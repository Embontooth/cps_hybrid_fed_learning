import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
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
    raise ValueError("Could not detect label column. Pass --label-col explicitly.")


def to_binary_labels(series: pd.Series, normal_label: str) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    normal_token = normal_label.strip().lower()
    y = (normalized != normal_token).astype(np.int64).to_numpy()
    return y


def load_and_prepare(csv_path: Path, label_col: str | None, normal_label: str, test_size: float, val_size: float, random_state: int, subset_fraction: float = 1.0):
    df = pd.read_csv(csv_path)
    
    # If subset_fraction < 1.0, sample that fraction of rows
    if subset_fraction < 1.0:
        rng = np.random.default_rng(random_state)
        sample_size = int(len(df) * subset_fraction)
        indices = rng.choice(len(df), size=sample_size, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
        print(f"Sampled {sample_size} rows ({subset_fraction*100:.1f}%) from {csv_path}")

    if label_col is None:
        label_col = find_label_column(df.columns)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in columns.")

    y = to_binary_labels(df[label_col], normal_label)

    X_df = df.drop(columns=[label_col]).copy()

    non_numeric = [col for col in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[col])]
    if non_numeric:
        X_df = X_df.drop(columns=non_numeric)

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    X = X_df.to_numpy(dtype=np.float32)

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_all,
    )

    X_train_normal = X_train_part[y_train_part == 0]
    X_val_normal = X_val[y_val == 0]

    if len(X_train_normal) == 0:
        raise ValueError("No normal samples found in training split. Check --normal-label.")
    if len(X_val_normal) == 0:
        raise ValueError("No normal samples found in validation split. Increase data size or adjust split.")

    scaler = StandardScaler()
    scaler.fit(X_train_normal)

    data = {
        "X_train_normal": scaler.transform(X_train_normal).astype(np.float32),
        "X_val_normal": scaler.transform(X_val_normal).astype(np.float32),
        "X_val_all": scaler.transform(X_val).astype(np.float32),
        "y_val": y_val,
        "X_test_all": scaler.transform(X_test).astype(np.float32),
        "y_test": y_test,
        "feature_names": X_df.columns.tolist(),
        "label_col": label_col,
    }
    return data, scaler


def build_loader(array: np.ndarray, batch_size: int, shuffle: bool):
    tensor = torch.from_numpy(array)
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def reconstruction_errors(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x).to(device)
        recon = model(tensor)
        mse = torch.mean((recon - tensor) ** 2, dim=1)
    return mse.cpu().numpy()


def train(model, train_loader, val_loader, device, epochs, lr, patience):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon = model(xb)
                loss = criterion(recon, xb)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))

        print(f"Epoch {epoch:03d} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


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

    return metrics, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train an autoencoder for CICIDS2017 anomaly detection.")
    parser.add_argument("--csv", type=str, default="cicids2017_cleaned.csv", help="Path to cleaned CSV file")
    parser.add_argument("--label-col", type=str, default=None, help="Label column name (auto-detected if omitted)")
    parser.add_argument("--normal-label", type=str, default="Normal Traffic", help="Value representing normal class")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--subset-fraction", type=float, default=1.0, help="Fraction of data to use (e.g., 0.2 for 1/5)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--threshold-quantile", type=float, default=0.99, help="Quantile on validation-normal reconstruction error")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, scaler = load_and_prepare(
        csv_path=Path(args.csv),
        label_col=args.label_col,
        normal_label=args.normal_label,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        subset_fraction=args.subset_fraction,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = data["X_train_normal"].shape[1]

    model = Autoencoder(input_dim=input_dim).to(device)

    train_loader = build_loader(data["X_train_normal"], args.batch_size, shuffle=True)
    val_loader = build_loader(data["X_val_normal"], args.batch_size, shuffle=False)

    train(model, train_loader, val_loader, device, args.epochs, args.lr, args.patience)

    val_normal_scores = reconstruction_errors(model, data["X_val_normal"], device)
    threshold = float(np.quantile(val_normal_scores, args.threshold_quantile))

    test_scores = reconstruction_errors(model, data["X_test_all"], device)
    metrics, y_pred = evaluate(test_scores, data["y_test"], threshold)

    print("\nEvaluation on test split:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification report:")
    print(classification_report(data["y_test"], y_pred, digits=4))

    model_path = output_dir / "autoencoder.pt"
    scaler_path = output_dir / "scaler.npz"
    meta_path = output_dir / "metadata.json"

    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_, features=np.array(data["feature_names"], dtype=object))

    metadata = {
        "label_column": data["label_col"],
        "normal_label": args.normal_label,
        "input_dim": input_dim,
        "threshold": threshold,
        "threshold_quantile": args.threshold_quantile,
        "metrics": metrics,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
