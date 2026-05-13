"""
Vertical Federated Learning (VFL) Split Learning Trainer for CICIDS2017

In VFL, each client processes only its subset of features and produces embeddings.
The server aggregates embeddings and computes the global loss.

Architecture:
  Client 0 (Fwd): 13 features → Embedding (32-dim)
  Client 1 (Bwd): 13 features → Embedding (32-dim)
  Client 2 (Flow): 26 features → Embedding (32-dim)
                                  ↓
  Server: Concat embeddings (96-dim) → Reconstruction (52 features)
          MSE Loss: ||X_reconstructed - X_original||²
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class ClientEmbeddingModel(nn.Module):
    """Local model at each VFL client that extracts embeddings from its features"""
    def __init__(self, input_dim: int, embedding_dim: int = 32):
        super().__init__()
        # Encoder: compress input features to embedding
        hidden_dim = max(16, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from input features"""
        return self.encoder(x)


class GlobalReconstructionModel(nn.Module):
    """Server model that aggregates embeddings and reconstructs original features"""
    def __init__(self, total_embedding_dim: int, output_dim: int):
        super().__init__()
        # Decoder: reconstruct features from aggregated embeddings
        hidden_dim = max(64, output_dim)
        self.decoder = nn.Sequential(
            nn.Linear(total_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, aggregated_embedding: torch.Tensor) -> torch.Tensor:
        """Reconstruct features from aggregated embedding"""
        return self.decoder(aggregated_embedding)


def find_label_column(columns):
    """Auto-detect label column"""
    preferred = ["Attack Type", "Label", "label", "attack", "attack_type"]
    for col in preferred:
        if col in columns:
            return col
    raise ValueError("Could not detect label column.")


def to_binary_labels(series: pd.Series, normal_label: str) -> np.ndarray:
    """Convert labels to binary (0=normal, 1=attack)"""
    normalized = series.astype(str).str.strip().str.lower()
    normal_token = normal_label.strip().lower()
    return (normalized != normal_token).astype(np.int64).to_numpy()


def prepare_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Clean and prepare features"""
    x_df = df.drop(columns=[label_col]).copy()
    non_numeric = [col for col in x_df.columns if not pd.api.types.is_numeric_dtype(x_df[col])]
    if non_numeric:
        x_df = x_df.drop(columns=non_numeric)
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    return x_df


def build_loader(array: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Create PyTorch DataLoader"""
    tensor = torch.from_numpy(array)
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_vfl_data(csv_path: Path, label_col: str, normal_label: str, test_size: float, val_size: float, random_state: int):
    """Load and prepare data for VFL training"""
    df = pd.read_csv(csv_path)
    
    if label_col not in df.columns:
        label_col = find_label_column(df.columns)
    
    y = to_binary_labels(df[label_col], normal_label)
    X_df = prepare_features(df, label_col)
    X = X_df.to_numpy(dtype=np.float32)
    
    # Split data
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=val_size, random_state=random_state, stratify=y_train_all
    )
    
    # Extract normal samples for threshold tuning
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]
    
    if len(X_train_normal) == 0 or len(X_val_normal) == 0:
        raise ValueError("No normal samples in train or val split")
    
    # Fit scaler on normal training data
    scaler = StandardScaler()
    scaler.fit(X_train_normal)
    
    # Scale all data
    X_train_normal = scaler.transform(X_train_normal).astype(np.float32)
    X_val_normal = scaler.transform(X_val_normal).astype(np.float32)
    X_val_all = scaler.transform(X_val).astype(np.float32)
    X_test_all = scaler.transform(X_test).astype(np.float32)
    
    return {
        "X_train_normal": X_train_normal,
        "X_val_normal": X_val_normal,
        "X_val_all": X_val_all,
        "y_val": y_val,
        "X_test_all": X_test_all,
        "y_test": y_test,
        "feature_names": X_df.columns.tolist(),
        "label_col": label_col,
    }, scaler


# VFL STRATEGY 1: SEMANTIC GROUPING (3 CLIENTS)
VFL_FEATURE_SETS = {
    "client_0_forward_metrics": [
        "Total Fwd Packets",
        "Total Length of Fwd Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Fwd Header Length",
        "Fwd Packets/s",
    ],
    "client_1_backward_metrics": [
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Bwd Header Length",
        "Bwd Packets/s",
    ],
    "client_2_flow_and_flags": [
        "Destination Port",
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Min Packet Length",
        "Max Packet Length",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "FIN Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "Average Packet Size",
        "Subflow Fwd Bytes",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "act_data_pkt_fwd",
        "min_seg_size_forward",
        "Active Mean",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Max",
        "Idle Min",
    ],
}


def partition_features(X: np.ndarray, feature_names: list[str]) -> Dict[str, np.ndarray]:
    """Partition features among VFL clients"""
    client_data = {}
    
    for client_name, features in VFL_FEATURE_SETS.items():
        # Get indices of features for this client
        indices = [feature_names.index(f) for f in features if f in feature_names]
        if not indices:
            raise ValueError(f"No matching features found for {client_name}")
        
        client_data[client_name] = X[:, indices].astype(np.float32)
        print(f"  {client_name}: {len(indices)} features")
    
    return client_data


def vfl_train_round(client_models: Dict[str, nn.Module], global_model: nn.Module, 
                    train_x: np.ndarray, train_data: Dict[str, np.ndarray], 
                    device: torch.device, batch_size: int, lr: float) -> float:
    """Single VFL training round with proper end-to-end training"""
    criterion = nn.MSELoss()
    
    # Create optimizers for each client and global model
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=lr)
        for name, model in client_models.items()
    }
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=lr)
    
    n_samples = len(train_x)
    n_batches = max(1, n_samples // batch_size)
    
    total_loss = 0.0
    
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        
        # Get batch slices for each client
        x_fwd_batch = torch.from_numpy(train_data["client_0_forward_metrics"][batch_start:batch_end]).to(device)
        x_bwd_batch = torch.from_numpy(train_data["client_1_backward_metrics"][batch_start:batch_end]).to(device)
        x_flow_batch = torch.from_numpy(train_data["client_2_flow_and_flags"][batch_start:batch_end]).to(device)
        x_orig_batch = torch.from_numpy(train_x[batch_start:batch_end]).to(device)
        
        # Set to training mode
        for model in client_models.values():
            model.train()
        global_model.train()
        
        # Forward: extract embeddings from each client
        emb_fwd = client_models["client_0_forward_metrics"](x_fwd_batch)
        emb_bwd = client_models["client_1_backward_metrics"](x_bwd_batch)
        emb_flow = client_models["client_2_flow_and_flags"](x_flow_batch)
        
        # Aggregate embeddings
        aggregated = torch.cat([emb_fwd, emb_bwd, emb_flow], dim=1)
        
        # Reconstruct
        reconstructed = global_model(aggregated)
        
        # Compute loss
        loss = criterion(reconstructed, x_orig_batch)
        
        # Backward
        for opt in optimizers.values():
            opt.zero_grad()
        global_optimizer.zero_grad()
        
        loss.backward()
        
        # Update
        for opt in optimizers.values():
            opt.step()
        global_optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / n_batches


def evaluate_vfl(client_models: Dict[str, nn.Module], global_model: nn.Module,
                 val_x_orig: np.ndarray, val_data: Dict[str, np.ndarray], y_val: np.ndarray,
                 device: torch.device, batch_size: int) -> dict:
    """Evaluate VFL model on validation set"""
    
    # Set to eval mode
    for model in client_models.values():
        model.eval()
    global_model.eval()
    
    criterion = nn.MSELoss(reduction='none')
    
    n_samples = len(val_x_orig)
    all_errors = []
    
    with torch.no_grad():
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            
            # Get batch slices
            x_fwd = torch.from_numpy(val_data["client_0_forward_metrics"][batch_start:batch_end]).to(device)
            x_bwd = torch.from_numpy(val_data["client_1_backward_metrics"][batch_start:batch_end]).to(device)
            x_flow = torch.from_numpy(val_data["client_2_flow_and_flags"][batch_start:batch_end]).to(device)
            x_orig = torch.from_numpy(val_x_orig[batch_start:batch_end]).to(device)
            
            # Extract embeddings
            emb_fwd = client_models["client_0_forward_metrics"](x_fwd)
            emb_bwd = client_models["client_1_backward_metrics"](x_bwd)
            emb_flow = client_models["client_2_flow_and_flags"](x_flow)
            
            # Aggregate
            aggregated = torch.cat([emb_fwd, emb_bwd, emb_flow], dim=1)
            
            # Reconstruct
            reconstructed = global_model(aggregated)
            
            # Compute per-sample error
            mse = torch.mean((reconstructed - x_orig) ** 2, dim=1)
            all_errors.append(mse.cpu().numpy())
    
    # Aggregate errors
    scores = np.hstack(all_errors)
    
    # Ensure y_val matches scores length
    if len(y_val) != len(scores):
        print(f"Warning: y_val length {len(y_val)} != scores length {len(scores)}")
        min_len = min(len(y_val), len(scores))
        y_val = y_val[:min_len]
        scores = scores[:min_len]
    
    # Compute threshold and predictions
    threshold = np.quantile(scores, 0.99)
    y_pred = (scores > threshold).astype(np.int64)
    
    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }
    
    if len(np.unique(y_val)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_val, scores))
        metrics["pr_auc"] = float(average_precision_score(y_val, scores))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train VFL model for CICIDS2017 anomaly detection.")
    parser.add_argument("--csv", type=str, default="cicids2017_cleaned.csv")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--normal-label", type=str, default="Normal Traffic")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension per client")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--local-epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--rounds", type=int, default=10, help="Number of VFL rounds")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="vfl_artifacts")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    data, scaler = load_vfl_data(
        csv_path=Path(args.csv),
        label_col=args.label_col,
        normal_label=args.normal_label,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )
    
    print("\nPartitioning features for VFL clients...")
    train_data = partition_features(data["X_train_normal"], data["feature_names"])
    val_data = partition_features(data["X_val_normal"], data["feature_names"])
    
    print("\nInitializing VFL models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize client models
    client_models = {}
    for client_name, features in VFL_FEATURE_SETS.items():
        input_dim = len([f for f in features if f in data["feature_names"]])
        client_models[client_name] = ClientEmbeddingModel(input_dim, args.embedding_dim).to(device)
    
    # Initialize global model
    total_embedding_dim = args.embedding_dim * len(client_models)
    output_dim = len(data["feature_names"])
    global_model = GlobalReconstructionModel(total_embedding_dim, output_dim).to(device)
    
    print(f"\nVFL Architecture:")
    print(f"  Clients: {len(client_models)}")
    print(f"  Embedding dimension per client: {args.embedding_dim}")
    print(f"  Total aggregated embedding: {total_embedding_dim}")
    print(f"  Output dimension: {output_dim}")
    
    print(f"\nTraining VFL for {args.rounds} rounds...")
    print("=" * 70)
    
    best_f1 = 0.0
    best_state = None
    
    for round_num in range(1, args.rounds + 1):
        print(f"\nRound {round_num}/{args.rounds}")
        
        # Train round
        loss = vfl_train_round(client_models, global_model, data["X_train_normal"], 
                               train_data, device, args.batch_size, args.lr)
        
        # Evaluate
        metrics = evaluate_vfl(client_models, global_model, data["X_val_normal"],
                              val_data, data["y_val"], device, args.batch_size)
        
        print(f"  Loss: {loss:.6f} | F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_state = {
                'client_models': {name: copy.deepcopy(model.state_dict()) for name, model in client_models.items()},
                'global_model': copy.deepcopy(global_model.state_dict()),
                'metrics': metrics,
                'round': round_num,
            }
    
    # Load best state
    if best_state:
        for client_name, state_dict in best_state['client_models'].items():
            client_models[client_name].load_state_dict(state_dict)
        global_model.load_state_dict(best_state['global_model'])
        best_metrics = best_state['metrics']
        print(f"\n✓ Loaded best model from round {best_state['round']}")
    
    # Save models
    print(f"\nSaving models to {output_dir}...")
    for client_name, model in client_models.items():
        model_path = output_dir / f"{client_name}_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ {model_path}")
    
    global_model_path = output_dir / "global_reconstruction_model.pt"
    torch.save(global_model.state_dict(), global_model_path)
    print(f"  ✓ {global_model_path}")
    
    # Save scaler and metadata
    scaler_path = output_dir / "scaler.npz"
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_, 
             features=np.array(data["feature_names"], dtype=object))
    
    metadata = {
        "strategy": "VFL Strategy 1 - Semantic Grouping",
        "approach": "split_learning",
        "n_clients": len(client_models),
        "embedding_dim": args.embedding_dim,
        "total_embedding_dim": total_embedding_dim,
        "output_dim": output_dim,
        "rounds": args.rounds,
        "best_round": best_state['round'] if best_state else 0,
        "best_metrics": best_metrics if best_state else {},
        "feature_sets": {k: v for k, v in VFL_FEATURE_SETS.items()},
        "scaler_path": str(scaler_path),
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ {scaler_path}")
    print(f"  ✓ {metadata_path}")
    
    print("\n" + "=" * 70)
    print("VFL Training Complete!")
    print(f"Best F1 Score: {best_metrics['f1']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
