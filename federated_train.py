import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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
    raise ValueError("Could not detect label column. Pass --label-col.")


def to_binary_labels(series: pd.Series, normal_label: str) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    normal_token = normal_label.strip().lower()
    return (normalized != normal_token).astype(np.int64).to_numpy()


def prepare_features(df: pd.DataFrame, label_col: str):
    x_df = df.drop(columns=[label_col]).copy()
    non_numeric = [col for col in x_df.columns if not pd.api.types.is_numeric_dtype(x_df[col])]
    if non_numeric:
        x_df = x_df.drop(columns=non_numeric)
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    return x_df


def build_loader(x: np.ndarray, batch_size: int, shuffle: bool):
    tensor = torch.from_numpy(x)
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def local_train(model: nn.Module, train_loader: DataLoader, device: torch.device, local_epochs: int, lr: float):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(local_epochs):
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()


def reconstruction_errors(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x).to(device)
        recon = model(tensor)
        mse = torch.mean((recon - tensor) ** 2, dim=1)
    return mse.cpu().numpy()


def evaluate(scores: np.ndarray, y_true: np.ndarray, threshold: float):
    y_pred = (scores > threshold).astype(np.int64)
    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def weighted_average_state_dict(state_dicts: list[dict], weights: list[int]):
    total = float(sum(weights))
    avg_state = {}
    for key in state_dicts[0].keys():
        weighted = None
        for sd, w in zip(state_dicts, weights):
            contrib = sd[key] * (w / total)
            weighted = contrib if weighted is None else (weighted + contrib)
        avg_state[key] = weighted
    return avg_state


def discover_clients(data_root: Path):
    clients = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("client_")])
    if not clients:
        raise ValueError(f"No client directories found in {data_root}")
    return clients


def load_client_dataset(client_dir: Path, label_col: str | None, normal_label: str):
    train_normal_df = pd.read_csv(client_dir / "train_normal.csv")
    val_df = pd.read_csv(client_dir / "val.csv")
    test_df = pd.read_csv(client_dir / "test.csv")

    if label_col is None:
        label_col = find_label_column(train_normal_df.columns)

    x_train_df = prepare_features(train_normal_df, label_col)
    x_val_df = prepare_features(val_df, label_col)
    x_test_df = prepare_features(test_df, label_col)

    # Align feature sets by training columns
    common_cols = [c for c in x_train_df.columns if c in x_val_df.columns and c in x_test_df.columns]
    x_train = x_train_df[common_cols].to_numpy(dtype=np.float32)
    x_val = x_val_df[common_cols].to_numpy(dtype=np.float32)
    x_test = x_test_df[common_cols].to_numpy(dtype=np.float32)

    y_val = to_binary_labels(val_df[label_col], normal_label)
    y_test = to_binary_labels(test_df[label_col], normal_label)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_s = scaler.transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)
    x_test_s = scaler.transform(x_test).astype(np.float32)

    val_normal_mask = y_val == 0
    if not np.any(val_normal_mask):
        raise ValueError(f"{client_dir.name} has no normal samples in val split.")

    return {
        "client": client_dir.name,
        "feature_names": common_cols,
        "x_train": x_train_s,
        "x_val": x_val_s,
        "y_val": y_val,
        "x_test": x_test_s,
        "y_test": y_test,
        "x_val_normal": x_val_s[val_normal_mask],
        "n_train": int(len(x_train_s)),
    }, label_col


def evaluate_global_model_on_clients(model: nn.Module, clients_data: list[dict], device: torch.device, threshold_quantile: float):
    per_client = []
    weighted = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0, "pr_auc": 0.0}
    total_weight = 0

    for client_data in clients_data:
        val_normal_scores = reconstruction_errors(model, client_data["x_val_normal"], device)
        threshold = float(np.quantile(val_normal_scores, threshold_quantile))

        test_scores = reconstruction_errors(model, client_data["x_test"], device)
        metrics = evaluate(test_scores, client_data["y_test"], threshold)

        item = {
            "client": client_data["client"],
            "n_train": client_data["n_train"],
            "metrics": metrics,
        }
        per_client.append(item)

        w = client_data["n_train"]
        total_weight += w
        for key in weighted.keys():
            value = metrics.get(key)
            if value is not None:
                weighted[key] += value * w

    weighted_avg = {k: (weighted[k] / total_weight if total_weight > 0 else None) for k in weighted.keys()}
    return per_client, weighted_avg


def main():
    parser = argparse.ArgumentParser(description="Federated autoencoder training with FedAvg on partitioned CICIDS clients.")
    parser.add_argument("--data-root", type=str, default="federated_data_stratified")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--normal-label", type=str, default="Normal Traffic")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients participating per round")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="federated_artifacts")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client_dirs = discover_clients(data_root)

    clients_data = []
    label_col = args.label_col
    for client_dir in client_dirs:
        client_data, detected_label_col = load_client_dataset(client_dir, label_col, args.normal_label)
        label_col = detected_label_col
        clients_data.append(client_data)

    input_dims = {len(c["feature_names"]) for c in clients_data}
    if len(input_dims) != 1:
        raise ValueError("Clients do not share a consistent input dimension.")

    input_dim = input_dims.pop()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = Autoencoder(input_dim=input_dim).to(device)

    history = []
    rng = np.random.default_rng(args.seed)

    n_clients = len(clients_data)
    n_sampled = max(1, int(np.ceil(n_clients * args.fraction_fit)))

    print(f"Clients discovered: {n_clients}")
    print(f"Input dim: {input_dim}")
    print(f"Using device: {device}")

    for round_idx in range(1, args.rounds + 1):
        sampled_indices = rng.choice(n_clients, size=n_sampled, replace=False)
        sampled_clients = [clients_data[i] for i in sampled_indices]

        local_states = []
        local_weights = []

        global_state = copy.deepcopy(global_model.state_dict())

        for client_data in sampled_clients:
            local_model = Autoencoder(input_dim=input_dim).to(device)
            local_model.load_state_dict(global_state)

            train_loader = build_loader(client_data["x_train"], args.batch_size, shuffle=True)
            local_train(local_model, train_loader, device, args.local_epochs, args.lr)

            local_states.append({k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()})
            local_weights.append(client_data["n_train"])

        new_state = weighted_average_state_dict(local_states, local_weights)
        global_model.load_state_dict(new_state)

        round_record = {
            "round": round_idx,
            "sampled_clients": [c["client"] for c in sampled_clients],
        }

        if args.eval_every > 0 and (round_idx % args.eval_every == 0 or round_idx == args.rounds):
            per_client, weighted_avg = evaluate_global_model_on_clients(
                global_model,
                clients_data,
                device,
                args.threshold_quantile,
            )
            round_record["weighted_metrics"] = weighted_avg
            round_record["per_client"] = per_client
            print(
                f"Round {round_idx:03d} | "
                f"f1={weighted_avg['f1']:.4f}, recall={weighted_avg['recall']:.4f}, pr_auc={weighted_avg['pr_auc']:.4f}"
            )
        else:
            print(f"Round {round_idx:03d} completed")

        history.append(round_record)

    final_per_client, final_weighted = evaluate_global_model_on_clients(
        global_model,
        clients_data,
        device,
        args.threshold_quantile,
    )

    model_path = output_dir / "global_autoencoder.pt"
    history_path = output_dir / "history.json"
    summary_path = output_dir / "final_summary.json"

    torch.save(global_model.state_dict(), model_path)

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "data_root": str(data_root),
        "label_column": label_col,
        "normal_label": args.normal_label,
        "input_dim": input_dim,
        "rounds": args.rounds,
        "fraction_fit": args.fraction_fit,
        "local_epochs": args.local_epochs,
        "threshold_quantile": args.threshold_quantile,
        "final_weighted_metrics": final_weighted,
        "final_per_client": final_per_client,
        "global_model_path": str(model_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFinal weighted metrics:")
    for key, value in final_weighted.items():
        print(f"{key}: {value}")

    print(f"\nSaved global model to: {model_path}")
    print(f"Saved round history to: {history_path}")
    print(f"Saved final summary to: {summary_path}")


if __name__ == "__main__":
    main()
