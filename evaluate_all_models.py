import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def find_label_column(columns):
    preferred = ["Attack Type", "Label", "label", "attack", "attack_type"]
    for col in preferred:
        if col in columns:
            return col
    lowered = {c.lower().strip(): c for c in columns}
    for key in ["attack type", "label", "attack", "class"]:
        if key in lowered:
            return lowered[key]
    raise ValueError("Could not detect label column.")


def to_binary_labels(series: pd.Series, normal_label: str) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    normal_token = normal_label.strip().lower()
    return (normalized != normal_token).astype(np.int64).to_numpy()


def prepare_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    x_df = df.drop(columns=[label_col]).copy()
    non_numeric = [col for col in x_df.columns if not pd.api.types.is_numeric_dtype(x_df[col])]
    if non_numeric:
        x_df = x_df.drop(columns=non_numeric)
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    return x_df


def reconstruction_errors(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x.astype(np.float32)).to(device)
        recon = model(tensor)
        mse = torch.mean((recon - tensor) ** 2, dim=1)
    return mse.cpu().numpy()


def evaluate(scores: np.ndarray, y_true: np.ndarray, threshold: float):
    y_pred = (scores > threshold).astype(np.int64)
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    return metrics


def load_scaler_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    mean = data["mean"].astype(np.float64)
    scale = data["scale"].astype(np.float64)
    features = [str(f) for f in data["features"].tolist()]
    safe_scale = np.where(scale == 0, 1.0, scale)
    return mean, safe_scale, features


def transform_with_scaler(x_df: pd.DataFrame, features: list[str], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    missing = [f for f in features if f not in x_df.columns]
    if missing:
        raise ValueError(f"Missing expected features: {missing[:5]} ...")
    x = x_df[features].to_numpy(dtype=np.float64)
    x = (x - mean) / scale
    return x.astype(np.float32)


def fit_simple_scaler(x: np.ndarray):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return mean, scale


def resolve_model_path(path_value: str, fallback_dir: Path) -> Path:
    p = Path(path_value)
    if p.exists():
        return p
    alt = fallback_dir / p.name
    if alt.exists():
        return alt
    return p


def derive_federated_model_name(summary_path: Path, fed_summary: dict) -> str:
    rounds = fed_summary.get("rounds")
    if rounds is not None:
        return f"federated_r{int(rounds)}"

    parent = summary_path.parent.name
    if parent.startswith("federated_artifacts_"):
        return parent.replace("federated_artifacts_", "federated_")
    return "federated_global"


@dataclass
class ModelEntry:
    name: str
    model: nn.Module
    input_dim: int
    threshold_quantile: float
    label_column: str
    normal_label: str
    scaler_mode: str  # 'fixed' or 'per_client_train_normal'
    scaler_payload: tuple | None  # (mean, scale, features) for fixed


def load_models(workspace: Path, federated_summary_paths: list[Path], centralized_dir: str = "artifacts"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: list[ModelEntry] = []

    centralized_meta_path = workspace / centralized_dir / "metadata.json"
    centralized_meta = json.loads(centralized_meta_path.read_text(encoding="utf-8"))
    cent_model_path = resolve_model_path(centralized_meta["model_path"], centralized_meta_path.parent)
    cent_scaler_path = resolve_model_path(centralized_meta["scaler_path"], centralized_meta_path.parent)
    cent_mean, cent_scale, cent_features = load_scaler_npz(cent_scaler_path)

    cent_model = Autoencoder(input_dim=int(centralized_meta["input_dim"]))
    cent_model.load_state_dict(torch.load(cent_model_path, map_location=device))
    cent_model.to(device)

    # Create descriptive model name
    model_name = "centralized"
    if "subset" in centralized_dir or "1_5" in centralized_dir or "1fifth" in centralized_dir.lower():
        model_name = "centralized (1/5)"

    models.append(
        ModelEntry(
            name=model_name,
            model=cent_model,
            input_dim=int(centralized_meta["input_dim"]),
            threshold_quantile=float(centralized_meta.get("threshold_quantile", 0.99)),
            label_column=centralized_meta["label_column"],
            normal_label=centralized_meta["normal_label"],
            scaler_mode="fixed",
            scaler_payload=(cent_mean, cent_scale, cent_features),
        )
    )

    for local_meta_path in sorted(workspace.glob("artifacts_client_*/metadata.json")):
        local_meta = json.loads(local_meta_path.read_text(encoding="utf-8"))
        model_name = local_meta_path.parent.name.replace("artifacts_", "")

        local_model_path = resolve_model_path(local_meta["model_path"], local_meta_path.parent)
        local_scaler_path = resolve_model_path(local_meta["scaler_path"], local_meta_path.parent)
        loc_mean, loc_scale, loc_features = load_scaler_npz(local_scaler_path)

        local_model = Autoencoder(input_dim=int(local_meta["input_dim"]))
        local_model.load_state_dict(torch.load(local_model_path, map_location=device))
        local_model.to(device)

        models.append(
            ModelEntry(
                name=model_name,
                model=local_model,
                input_dim=int(local_meta["input_dim"]),
                threshold_quantile=float(local_meta.get("threshold_quantile", 0.99)),
                label_column=local_meta["label_column"],
                normal_label=local_meta["normal_label"],
                scaler_mode="fixed",
                scaler_payload=(loc_mean, loc_scale, loc_features),
            )
        )

    for federated_summary_path in federated_summary_paths:
        fed_summary = json.loads(federated_summary_path.read_text(encoding="utf-8"))
        fed_model_path = resolve_model_path(fed_summary["global_model_path"], federated_summary_path.parent)

        fed_model = Autoencoder(input_dim=int(fed_summary["input_dim"]))
        fed_model.load_state_dict(torch.load(fed_model_path, map_location=device))
        fed_model.to(device)

        models.append(
            ModelEntry(
                name=derive_federated_model_name(federated_summary_path, fed_summary),
                model=fed_model,
                input_dim=int(fed_summary["input_dim"]),
                threshold_quantile=float(fed_summary.get("threshold_quantile", 0.99)),
                label_column=fed_summary["label_column"],
                normal_label=fed_summary["normal_label"],
                scaler_mode="per_client_train_normal",
                scaler_payload=None,
            )
        )

    return models, device


def evaluate_on_clients(models: list[ModelEntry], device: torch.device, data_root: Path, output_dir: Path):
    client_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("client_")])
    if not client_dirs:
        raise ValueError(f"No client folders found in {data_root}")

    rows = []

    for model_entry in models:
        for client_dir in client_dirs:
            train_normal_df = pd.read_csv(client_dir / "train_normal.csv")
            val_df = pd.read_csv(client_dir / "val.csv")
            test_df = pd.read_csv(client_dir / "test.csv")

            label_col = model_entry.label_column if model_entry.label_column in val_df.columns else find_label_column(val_df.columns)

            x_train_normal_df = prepare_features(train_normal_df, label_col)
            x_val_df = prepare_features(val_df, label_col)
            x_test_df = prepare_features(test_df, label_col)

            y_val = to_binary_labels(val_df[label_col], model_entry.normal_label)
            y_test = to_binary_labels(test_df[label_col], model_entry.normal_label)

            if model_entry.scaler_mode == "fixed":
                mean, scale, features = model_entry.scaler_payload
            else:
                common_features = [c for c in x_train_normal_df.columns if c in x_val_df.columns and c in x_test_df.columns]
                x_train_for_scaler = x_train_normal_df[common_features].to_numpy(dtype=np.float64)
                mean, scale = fit_simple_scaler(x_train_for_scaler)
                features = common_features

            x_val = transform_with_scaler(x_val_df, features, mean, scale)
            x_test = transform_with_scaler(x_test_df, features, mean, scale)

            val_normal_mask = y_val == 0
            if not np.any(val_normal_mask):
                continue

            val_normal_scores = reconstruction_errors(model_entry.model, x_val[val_normal_mask], device)
            threshold = float(np.quantile(val_normal_scores, model_entry.threshold_quantile))

            test_scores = reconstruction_errors(model_entry.model, x_test, device)
            metrics = evaluate(test_scores, y_test, threshold)

            rows.append(
                {
                    "model": model_entry.name,
                    "client_test": client_dir.name,
                    "n_test": int(len(y_test)),
                    "threshold": threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                }
            )

    detail_df = pd.DataFrame(rows)
    detail_csv = output_dir / "comparison_detail.csv"
    detail_df.to_csv(detail_csv, index=False)

    weighted_rows = []
    for model_name, grp in detail_df.groupby("model"):
        w = grp["n_test"].to_numpy(dtype=np.float64)

        def wavg(col: str):
            values = grp[col].to_numpy(dtype=np.float64)
            mask = ~np.isnan(values)
            if not np.any(mask):
                return np.nan
            return float(np.average(values[mask], weights=w[mask]))

        weighted_rows.append(
            {
                "model": model_name,
                "weighted_precision": wavg("precision"),
                "weighted_recall": wavg("recall"),
                "weighted_f1": wavg("f1"),
                "weighted_roc_auc": wavg("roc_auc"),
                "weighted_pr_auc": wavg("pr_auc"),
            }
        )

    summary_df = pd.DataFrame(weighted_rows).sort_values(by="weighted_f1", ascending=False)
    summary_csv = output_dir / "comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    return detail_df, summary_df, detail_csv, summary_csv


def save_graphs(detail_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    metrics_to_plot = ["weighted_precision", "weighted_recall", "weighted_f1", "weighted_pr_auc", "weighted_roc_auc"]
    metric_labels = {
        "weighted_precision": "Precision",
        "weighted_recall": "Recall",
        "weighted_f1": "F1",
        "weighted_pr_auc": "PR-AUC",
        "weighted_roc_auc": "ROC-AUC",
    }

    ordered = summary_df.sort_values(by="weighted_f1", ascending=False).reset_index(drop=True)
    model_names = ordered["model"].tolist()

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / max(1, len(model_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(model_names):
        row = ordered[ordered["model"] == model_name].iloc[0]
        vals = [row[m] for m in metrics_to_plot]
        ax.bar(x + i * width, vals, width=width, label=model_name)

    ax.set_title("Weighted Metric Comparison Across Models")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x + (len(model_names) - 1) * width / 2)
    ax.set_xticklabels([metric_labels[m] for m in metrics_to_plot])
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    weighted_plot_path = output_dir / "weighted_metrics_bar.png"
    fig.savefig(weighted_plot_path, dpi=150)
    plt.close(fig)

    pivot_f1 = detail_df.pivot(index="model", columns="client_test", values="f1")
    pivot_f1 = pivot_f1.reindex(model_names)

    return weighted_plot_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate centralized, federated, and local models on identical client test files.")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--data-root", type=str, default="federated_data_stratified")
    parser.add_argument("--federated-summary", type=str, default="federated_artifacts_r20/final_summary.json")
    parser.add_argument(
        "--federated-summaries",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of federated summary paths (e.g. r20 r40 r80 r100).",
    )
    parser.add_argument("--centralized-dir", type=str, default="artifacts", help="Directory containing centralized model")
    parser.add_argument("--output-dir", type=str, default="evaluation_comparison")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    data_root = workspace / args.data_root
    if args.federated_summaries:
        fed_summaries = [workspace / p for p in args.federated_summaries]
    else:
        fed_summaries = [workspace / args.federated_summary]

    missing = [str(p) for p in fed_summaries if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Federated summary files not found: {missing}")

    output_dir = workspace / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models, device = load_models(workspace, fed_summaries, args.centralized_dir)
    detail_df, summary_df, detail_csv, summary_csv = evaluate_on_clients(models, device, data_root, output_dir)
    weighted_plot_path = save_graphs(detail_df, summary_df, output_dir)

    pd.options.display.float_format = "{:.6f}".format
    print("\n=== Weighted Comparison (same client test files) ===")
    print(summary_df.to_string(index=False))

    print("\n=== Per-client Detail (first 20 rows) ===")
    print(detail_df.head(20).to_string(index=False))

    print(f"\nSaved detailed table to: {detail_csv}")
    print(f"Saved summary table to: {summary_csv}")
    print(f"Saved weighted metrics graph to: {weighted_plot_path}")

if __name__ == "__main__":
    main()
