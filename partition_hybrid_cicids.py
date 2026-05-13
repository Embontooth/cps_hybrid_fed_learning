import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vfl_train_corrected import VFL_FEATURE_SETS, find_label_column, to_binary_labels


def dirichlet_partition_indices(y: np.ndarray, n_orgs: int, alpha: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    org_indices = [[] for _ in range(n_orgs)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(np.repeat(alpha, n_orgs))
        split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, split_points)

        for org_id, split in enumerate(splits):
            org_indices[org_id].extend(split.tolist())

    out = []
    for idx in org_indices:
        arr = np.array(idx, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def stratified_partition_indices(y: np.ndarray, n_orgs: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    org_indices = [[] for _ in range(n_orgs)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        splits = np.array_split(cls_idx, n_orgs)
        for org_id, split in enumerate(splits):
            org_indices[org_id].extend(split.tolist())

    out = []
    for idx in org_indices:
        arr = np.array(idx, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def safe_split(df: pd.DataFrame, y_binary: np.ndarray, test_size: float, val_size: float, seed: int):
    can_stratify = len(np.unique(y_binary)) > 1 and len(df) >= 10

    if can_stratify:
        train_df, test_df, y_train, y_test = train_test_split(
            df, y_binary, test_size=test_size, random_state=seed, stratify=y_binary
        )
    else:
        train_df, test_df, y_train, y_test = train_test_split(
            df, y_binary, test_size=test_size, random_state=seed, stratify=None
        )

    can_stratify_val = len(np.unique(y_train)) > 1 and len(train_df) >= 10
    if can_stratify_val:
        train_df, val_df, y_train2, y_val = train_test_split(
            train_df, y_train, test_size=val_size, random_state=seed, stratify=y_train
        )
    else:
        train_df, val_df, y_train2, y_val = train_test_split(
            train_df, y_train, test_size=val_size, random_state=seed, stratify=None
        )

    return train_df, val_df, test_df, y_train2, y_val, y_test


def summarize_split(y_binary: np.ndarray) -> dict:
    total = int(len(y_binary))
    anomaly = int(np.sum(y_binary == 1))
    normal = int(np.sum(y_binary == 0))
    return {
        "rows": total,
        "normal": normal,
        "anomaly": anomaly,
        "anomaly_ratio": float(anomaly / total) if total else 0.0,
    }


def build_feature_indices(feature_names: list[str]) -> dict[str, list[int]]:
    indices = {}
    for client_name, features in VFL_FEATURE_SETS.items():
        idx = [feature_names.index(f) for f in features if f in feature_names]
        if not idx:
            raise ValueError(f"No feature matches found for {client_name}")
        indices[client_name] = idx
    return indices


def main():
    parser = argparse.ArgumentParser(
        description="Create hybrid VHFL data splits: horizontal rows by organization + shared vertical feature metadata."
    )
    parser.add_argument("--csv", type=str, default="cicids2017_cleaned.csv")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--normal-label", type=str, default="Normal Traffic")
    parser.add_argument("--num-orgs", type=int, default=5)
    parser.add_argument("--strategy", choices=["dirichlet", "stratified"], default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="hybrid_data")
    args = parser.parse_args()

    input_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    label_col = args.label_col or find_label_column(df.columns)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in input CSV.")

    y_binary = to_binary_labels(df[label_col], args.normal_label)

    feature_names = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    feature_indices = build_feature_indices(feature_names)

    if args.strategy == "dirichlet":
        org_splits = dirichlet_partition_indices(y_binary, args.num_orgs, args.alpha, args.seed)
    else:
        org_splits = stratified_partition_indices(y_binary, args.num_orgs, args.seed)

    summary = {
        "source_csv": str(input_path),
        "label_col": label_col,
        "normal_label": args.normal_label,
        "num_orgs": args.num_orgs,
        "strategy": args.strategy,
        "alpha": args.alpha if args.strategy == "dirichlet" else None,
        "splits": {"test_size": args.test_size, "val_size": args.val_size},
        "vfl_feature_sets": VFL_FEATURE_SETS,
        "feature_names": feature_names,
        "vfl_feature_indices": feature_indices,
        "organizations": [],
    }

    for org_id, indices in enumerate(org_splits):
        org_name = f"org_{org_id:02d}"
        org_dir = output_dir / org_name
        org_dir.mkdir(parents=True, exist_ok=True)

        org_df = df.iloc[indices].copy()
        org_y = y_binary[indices]

        train_df, val_df, test_df, y_train, y_val, y_test = safe_split(
            org_df, org_y, test_size=args.test_size, val_size=args.val_size, seed=args.seed
        )

        train_normal_df = train_df[y_train == 0].copy()

        train_df.to_csv(org_dir / "train.csv", index=False)
        train_normal_df.to_csv(org_dir / "train_normal.csv", index=False)
        val_df.to_csv(org_dir / "val.csv", index=False)
        test_df.to_csv(org_dir / "test.csv", index=False)

        org_stats = {
            "org": org_name,
            "full": summarize_split(org_y),
            "train": summarize_split(y_train),
            "train_normal_rows": int(len(train_normal_df)),
            "val": summarize_split(y_val),
            "test": summarize_split(y_test),
        }
        summary["organizations"].append(org_stats)

    summary_path = output_dir / "partition_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved hybrid partitions to: {output_dir}")
    print(f"Summary: {summary_path}")
    for org in summary["organizations"]:
        print(
            f"{org['org']}: rows={org['full']['rows']}, "
            f"anomaly_ratio={org['full']['anomaly_ratio']:.4f}, train_normal={org['train_normal_rows']}"
        )


if __name__ == "__main__":
    main()
