import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def detect_label_column(columns):
    preferred = ["Attack Type", "Label", "label", "attack", "attack_type"]
    for col in preferred:
        if col in columns:
            return col
    lowered = {c.lower().strip(): c for c in columns}
    for key in ["attack type", "label", "attack", "class"]:
        if key in lowered:
            return lowered[key]
    raise ValueError("Could not detect label column. Use --label-col.")


def build_binary_label(series: pd.Series, normal_label: str) -> np.ndarray:
    return (series.astype(str).str.strip().str.lower() != normal_label.strip().lower()).astype(np.int64).to_numpy()


def dirichlet_partition_indices(y: np.ndarray, n_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(n_clients)]

    classes = np.unique(y)
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        cls_splits = np.split(cls_idx, split_points)

        for client_id, split in enumerate(cls_splits):
            client_indices[client_id].extend(split.tolist())

    out = []
    for idx in client_indices:
        arr = np.array(idx, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def stratified_partition_indices(y: np.ndarray, n_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(n_clients)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        splits = np.array_split(cls_idx, n_clients)
        for cid, s in enumerate(splits):
            client_indices[cid].extend(s.tolist())

    out = []
    for idx in client_indices:
        arr = np.array(idx, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def safe_train_val_test_split(
    df: pd.DataFrame,
    y_binary: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
):
    # For tiny or single-class client slices, fallback to unstratified splitting.
    can_stratify = len(np.unique(y_binary)) > 1 and len(df) >= 10

    if can_stratify:
        train_df, test_df, y_train, y_test = train_test_split(
            df,
            y_binary,
            test_size=test_size,
            random_state=seed,
            stratify=y_binary,
        )
    else:
        train_df, test_df, y_train, y_test = train_test_split(
            df,
            y_binary,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    can_stratify_val = len(np.unique(y_train)) > 1 and len(train_df) >= 10
    if can_stratify_val:
        train_df, val_df, y_train2, y_val = train_test_split(
            train_df,
            y_train,
            test_size=val_size,
            random_state=seed,
            stratify=y_train,
        )
    else:
        train_df, val_df, y_train2, y_val = train_test_split(
            train_df,
            y_train,
            test_size=val_size,
            random_state=seed,
            stratify=None,
        )

    return train_df, val_df, test_df, y_train2, y_val, y_test


def summarize_split(y_binary: np.ndarray) -> dict:
    total = int(len(y_binary))
    anomalies = int(np.sum(y_binary == 1))
    normals = int(np.sum(y_binary == 0))
    ratio = float(anomalies / total) if total else 0.0
    return {
        "rows": total,
        "normal": normals,
        "anomaly": anomalies,
        "anomaly_ratio": ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Partition CICIDS dataset into federated client splits.")
    parser.add_argument("--csv", type=str, default="cicids2017_cleaned.csv")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--normal-label", type=str, default="Normal Traffic")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--strategy", choices=["dirichlet", "stratified"], default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration (lower = more non-IID)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="federated_data")
    args = parser.parse_args()

    input_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    label_col = args.label_col or detect_label_column(df.columns)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    y_binary_all = build_binary_label(df[label_col], args.normal_label)

    if args.strategy == "dirichlet":
        client_indices = dirichlet_partition_indices(y_binary_all, args.num_clients, args.alpha, args.seed)
    else:
        client_indices = stratified_partition_indices(y_binary_all, args.num_clients, args.seed)

    summary = {
        "source_csv": str(input_path),
        "label_col": label_col,
        "normal_label": args.normal_label,
        "num_clients": args.num_clients,
        "strategy": args.strategy,
        "alpha": args.alpha if args.strategy == "dirichlet" else None,
        "splits": {
            "test_size": args.test_size,
            "val_size": args.val_size,
        },
        "clients": [],
    }

    for client_id, indices in enumerate(client_indices):
        client_name = f"client_{client_id:02d}"
        client_dir = output_dir / client_name
        client_dir.mkdir(parents=True, exist_ok=True)

        client_df = df.iloc[indices].copy()
        client_y = y_binary_all[indices]

        train_df, val_df, test_df, y_train, y_val, y_test = safe_train_val_test_split(
            client_df,
            client_y,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )

        train_normal_df = train_df[y_train == 0].copy()

        train_df.to_csv(client_dir / "train.csv", index=False)
        train_normal_df.to_csv(client_dir / "train_normal.csv", index=False)
        val_df.to_csv(client_dir / "val.csv", index=False)
        test_df.to_csv(client_dir / "test.csv", index=False)

        client_stats = {
            "client": client_name,
            "full": summarize_split(client_y),
            "train": summarize_split(y_train),
            "train_normal_rows": int(len(train_normal_df)),
            "val": summarize_split(y_val),
            "test": summarize_split(y_test),
        }
        summary["clients"].append(client_stats)

    summary_path = output_dir / "partition_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved partitioned data to: {output_dir}")
    print(f"Summary: {summary_path}")
    for c in summary["clients"]:
        print(
            f"{c['client']}: rows={c['full']['rows']}, anomaly_ratio={c['full']['anomaly_ratio']:.4f}, "
            f"train_normal={c['train_normal_rows']}"
        )


if __name__ == "__main__":
    main()
