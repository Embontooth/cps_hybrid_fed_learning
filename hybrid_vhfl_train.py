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

from vfl_train_corrected import (
    ClientEmbeddingModel,
    GlobalReconstructionModel,
    VFL_FEATURE_SETS,
    find_label_column,
    prepare_features,
    to_binary_labels,
)


def fedavg_state_dicts(state_dicts: list[dict], weights: list[int]) -> dict:
    total = float(sum(weights))
    averaged = {}
    for key in state_dicts[0].keys():
        accum = None
        for sd, w in zip(state_dicts, weights):
            contrib = sd[key] * (w / total)
            accum = contrib if accum is None else (accum + contrib)
        averaged[key] = accum
    return averaged


def evaluate_scores(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
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


def partition_feature_indices(feature_names: list[str]) -> dict[str, list[int]]:
    indices = {}
    for client_name, features in VFL_FEATURE_SETS.items():
        idx = [feature_names.index(f) for f in features if f in feature_names]
        if not idx:
            raise ValueError(f"No features matched for {client_name}")
        indices[client_name] = idx
    return indices


def load_org_data(org_dir: Path, label_col: str | None, normal_label: str):
    train_normal_df = pd.read_csv(org_dir / "train_normal.csv")
    val_df = pd.read_csv(org_dir / "val.csv")
    test_df = pd.read_csv(org_dir / "test.csv")

    if label_col is None:
        label_col = find_label_column(train_normal_df.columns)

    x_train_df = prepare_features(train_normal_df, label_col)
    x_val_df = prepare_features(val_df, label_col)
    x_test_df = prepare_features(test_df, label_col)

    feature_names = [c for c in x_train_df.columns if c in x_val_df.columns and c in x_test_df.columns]
    x_train = x_train_df[feature_names].to_numpy(dtype=np.float32)
    x_val = x_val_df[feature_names].to_numpy(dtype=np.float32)
    x_test = x_test_df[feature_names].to_numpy(dtype=np.float32)

    y_val = to_binary_labels(val_df[label_col], normal_label)
    y_test = to_binary_labels(test_df[label_col], normal_label)

    scaler = StandardScaler()
    scaler.fit(x_train)

    org = {
        "org": org_dir.name,
        "feature_names": feature_names,
        "x_train_normal": scaler.transform(x_train).astype(np.float32),
        "x_val": scaler.transform(x_val).astype(np.float32),
        "y_val": y_val,
        "x_test": scaler.transform(x_test).astype(np.float32),
        "y_test": y_test,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float64),
    }
    return org, label_col


def local_vhfl_update(
    global_client_models: dict[str, nn.Module],
    global_decoder: nn.Module,
    org_data: dict,
    feature_indices: dict[str, list[int]],
    device: torch.device,
    batch_size: int,
    lr: float,
    local_epochs: int,
) -> tuple[dict[str, dict], dict, float]:
    client_models = {}
    for name, model in global_client_models.items():
        m = copy.deepcopy(model).to(device)
        m.train()
        client_models[name] = m

    decoder = copy.deepcopy(global_decoder).to(device)
    decoder.train()

    optimizers = {name: torch.optim.Adam(model.parameters(), lr=lr) for name, model in client_models.items()}
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_full = org_data["x_train_normal"]
    n_samples = len(x_full)
    if n_samples == 0:
        raise ValueError(f"{org_data['org']} has no normal training rows.")

    total_loss = 0.0
    batches_seen = 0
    client_order = list(feature_indices.keys())

    for _ in range(local_epochs):
        perm = np.random.permutation(n_samples)
        shuffled = x_full[perm]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch_np = shuffled[start:end]

            x_batch = torch.from_numpy(x_batch_np).to(device)
            embeddings = []
            for client_name in client_order:
                idx = feature_indices[client_name]
                x_part = torch.from_numpy(x_batch_np[:, idx]).to(device)
                embeddings.append(client_models[client_name](x_part))

            aggregated = torch.cat(embeddings, dim=1)
            reconstructed = decoder(aggregated)
            loss = criterion(reconstructed, x_batch)

            for opt in optimizers.values():
                opt.zero_grad()
            decoder_optimizer.zero_grad()

            loss.backward()

            for opt in optimizers.values():
                opt.step()
            decoder_optimizer.step()

            total_loss += float(loss.item())
            batches_seen += 1

    client_state = {
        name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for name, model in client_models.items()
    }
    decoder_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    avg_loss = total_loss / max(1, batches_seen)
    return client_state, decoder_state, avg_loss


def score_org(
    client_models: dict[str, nn.Module],
    decoder: nn.Module,
    org_data: dict,
    feature_indices: dict[str, list[int]],
    threshold_quantile: float,
    device: torch.device,
) -> dict:
    for m in client_models.values():
        m.eval()
    decoder.eval()

    client_order = list(feature_indices.keys())

    def compute_scores(x_full_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_full = torch.from_numpy(x_full_np).to(device)
            embeddings = []
            for client_name in client_order:
                idx = feature_indices[client_name]
                x_part = torch.from_numpy(x_full_np[:, idx]).to(device)
                embeddings.append(client_models[client_name](x_part))
            aggregated = torch.cat(embeddings, dim=1)
            recon = decoder(aggregated)
            mse = torch.mean((recon - x_full) ** 2, dim=1)
            return mse.cpu().numpy()

    val_scores = compute_scores(org_data["x_val"])
    val_normal_mask = org_data["y_val"] == 0
    if np.any(val_normal_mask):
        threshold = float(np.quantile(val_scores[val_normal_mask], threshold_quantile))
    else:
        threshold = float(np.quantile(val_scores, threshold_quantile))

    test_scores = compute_scores(org_data["x_test"])
    metrics = evaluate_scores(test_scores, org_data["y_test"], threshold)
    return metrics


def evaluate_global(
    global_client_models: dict[str, nn.Module],
    global_decoder: nn.Module,
    orgs_data: list[dict],
    feature_indices: dict[str, list[int]],
    threshold_quantile: float,
    device: torch.device,
):
    per_org = []
    weighted = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0, "pr_auc": 0.0, "threshold": 0.0}
    total_weight = 0

    for org in orgs_data:
        metrics = score_org(global_client_models, global_decoder, org, feature_indices, threshold_quantile, device)
        per_org.append({"org": org["org"], "n_train": org["n_train"], "n_test": org["n_test"], "metrics": metrics})

        w = org["n_test"]
        total_weight += w
        for key in weighted:
            value = metrics.get(key)
            if value is not None:
                weighted[key] += value * w

    weighted_avg = {k: (weighted[k] / total_weight if total_weight > 0 else None) for k in weighted}
    return per_org, weighted_avg


def main():
    parser = argparse.ArgumentParser(
        description="Train vertical-horizontal hybrid FL (VHFL): VFL split-learning inside orgs + FedAvg across orgs."
    )
    parser.add_argument("--data-root", type=str, default="hybrid_data")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--normal-label", type=str, default="Normal Traffic")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="hybrid_artifacts")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    org_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("org_")])
    if not org_dirs:
        raise ValueError(f"No organization directories found in {data_root}")

    orgs_data = []
    label_col = args.label_col
    for org_dir in org_dirs:
        org_data, detected = load_org_data(org_dir, label_col, args.normal_label)
        label_col = detected
        orgs_data.append(org_data)

    feature_name_sets = {tuple(org["feature_names"]) for org in orgs_data}
    if len(feature_name_sets) != 1:
        raise ValueError("Organizations do not share the same feature ordering.")

    feature_names = orgs_data[0]["feature_names"]
    feature_indices = partition_feature_indices(feature_names)

    input_dim = len(feature_names)
    client_order = list(feature_indices.keys())
    total_embedding_dim = args.embedding_dim * len(client_order)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_client_models = {}
    for client_name in client_order:
        part_dim = len(feature_indices[client_name])
        global_client_models[client_name] = ClientEmbeddingModel(part_dim, args.embedding_dim).to(device)

    global_decoder = GlobalReconstructionModel(total_embedding_dim, input_dim).to(device)

    history = []

    print(f"Organizations discovered: {len(orgs_data)}")
    print(f"Using device: {device}")
    print(f"Input dim: {input_dim} | Embedding dim per vertical client: {args.embedding_dim}")

    for round_idx in range(1, args.rounds + 1):
        org_client_states = {name: [] for name in client_order}
        org_decoder_states = []
        org_weights = []
        round_losses = []

        for org_data in orgs_data:
            client_state, decoder_state, avg_loss = local_vhfl_update(
                global_client_models=global_client_models,
                global_decoder=global_decoder,
                org_data=org_data,
                feature_indices=feature_indices,
                device=device,
                batch_size=args.batch_size,
                lr=args.lr,
                local_epochs=args.local_epochs,
            )

            round_losses.append(avg_loss)
            org_weights.append(org_data["n_train"])
            for client_name in client_order:
                org_client_states[client_name].append(client_state[client_name])
            org_decoder_states.append(decoder_state)

        for client_name in client_order:
            averaged = fedavg_state_dicts(org_client_states[client_name], org_weights)
            global_client_models[client_name].load_state_dict(averaged)

        averaged_decoder = fedavg_state_dicts(org_decoder_states, org_weights)
        global_decoder.load_state_dict(averaged_decoder)

        round_record = {
            "round": round_idx,
            "local_loss_mean": float(np.mean(round_losses)),
            "local_loss_std": float(np.std(round_losses)),
        }

        if args.eval_every > 0 and (round_idx % args.eval_every == 0 or round_idx == args.rounds):
            per_org, weighted = evaluate_global(
                global_client_models,
                global_decoder,
                orgs_data,
                feature_indices,
                args.threshold_quantile,
                device,
            )
            round_record["weighted_metrics"] = weighted
            round_record["per_org"] = per_org
            print(
                f"Round {round_idx:03d} | loss={round_record['local_loss_mean']:.6f} "
                f"| f1={weighted['f1']:.4f} | recall={weighted['recall']:.4f} | pr_auc={weighted['pr_auc']:.4f}"
            )
        else:
            print(f"Round {round_idx:03d} | loss={round_record['local_loss_mean']:.6f}")

        history.append(round_record)

    final_per_org, final_weighted = evaluate_global(
        global_client_models,
        global_decoder,
        orgs_data,
        feature_indices,
        args.threshold_quantile,
        device,
    )

    for client_name in client_order:
        torch.save(global_client_models[client_name].state_dict(), output_dir / f"{client_name}_model.pt")
    torch.save(global_decoder.state_dict(), output_dir / "global_reconstruction_model.pt")

    scaler_means = np.vstack([org["scaler_mean"] for org in orgs_data])
    scaler_scales = np.vstack([org["scaler_scale"] for org in orgs_data])
    np.savez(
        output_dir / "scaler.npz",
        orgs=np.array([org["org"] for org in orgs_data], dtype=object),
        features=np.array(feature_names, dtype=object),
        means=scaler_means,
        scales=scaler_scales,
    )

    metadata = {
        "approach": "vhfl_split_learning_fedavg",
        "data_root": str(data_root),
        "label_column": label_col,
        "normal_label": args.normal_label,
        "n_organizations": len(orgs_data),
        "vertical_clients": client_order,
        "embedding_dim": args.embedding_dim,
        "total_embedding_dim": total_embedding_dim,
        "input_dim": input_dim,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "threshold_quantile": args.threshold_quantile,
        "feature_sets": VFL_FEATURE_SETS,
        "feature_indices": feature_indices,
    }

    summary = {
        "approach": "vhfl_split_learning_fedavg",
        "data_root": str(data_root),
        "global_model_path": str(output_dir / "global_reconstruction_model.pt"),
        "label_column": label_col,
        "normal_label": args.normal_label,
        "input_dim": input_dim,
        "embedding_dim": args.embedding_dim,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "threshold_quantile": args.threshold_quantile,
        "final_weighted_metrics": final_weighted,
        "final_per_org": final_per_org,
    }

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFinal weighted metrics:")
    for k, v in final_weighted.items():
        print(f"{k}: {v}")

    print(f"\nSaved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
