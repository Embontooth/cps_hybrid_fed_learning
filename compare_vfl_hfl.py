#!/usr/bin/env python3
"""
Compare VFL vs HFL vs Centralized Models on Test Set

This script loads the best models from each approach and evaluates them
side-by-side on the CICIDS2017 test set.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, roc_curve, 
    precision_recall_curve, auc
)
import argparse

from vfl_train_corrected import VFL_FEATURE_SETS


class ClientEmbeddingModel(nn.Module):
    """Local model at each VFL client that extracts embeddings from its features"""
    def __init__(self, input_dim: int, embedding_dim: int = 32):
        super().__init__()
        hidden_dim = max(16, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.encoder(x)


class GlobalReconstructionModel(nn.Module):
    """Server model that aggregates embeddings and reconstructs original features"""
    def __init__(self, total_embedding_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = max(64, output_dim)
        self.decoder = nn.Sequential(
            nn.Linear(total_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, aggregated_embedding):
        return self.decoder(aggregated_embedding)


class SimpleAutoencoder(nn.Module):
    """Standard autoencoder for centralized/federated models"""
    def __init__(self, input_dim=52):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def compute_reconstruction_metrics(y_true, scores, threshold):
    y_pred = (scores > threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, scores)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall_curve, precision_curve)
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_pred': y_pred,
    }


def get_feature_partition_indices(feature_names):
    indices = {}
    for client_name, features in VFL_FEATURE_SETS.items():
        idx = [feature_names.index(f) for f in features if f in feature_names]
        if not idx:
            raise ValueError(f"No feature mapping found for {client_name}")
        indices[client_name] = idx
    return indices


def load_data():
    """Load test data from partitioned dataset"""
    print("Loading test data...")
    
    # Load from first partition (representative of distribution)
    test_df = pd.read_csv('federated_data_stratified/client_00/test.csv')
    
    X_test = test_df.drop(columns=['Attack Type']).values.astype(np.float32)
    y_test = (test_df['Attack Type'] != 'Normal Traffic').astype(int).values
    
    # Normalize using the scaler from artifacts
    with np.load('artifacts/scaler.npz', allow_pickle=True) as data:
        mean = data['mean']
        scale = data['scale']
    
    X_test = (X_test - mean) / (scale + 1e-8)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels distribution: Normal={sum(y_test==0)}, Attack={sum(y_test==1)}")
    
    return X_test, y_test


def evaluate_centralized(X_test, y_test):
    """Evaluate centralized model"""
    print("\n" + "="*60)
    print("CENTRALIZED MODEL EVALUATION")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load best full-data model
    model = SimpleAutoencoder()
    checkpoint = torch.load('artifacts/autoencoder.pt', map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load threshold
    with open('artifacts/metadata.json', 'r') as f:
        metadata = json.load(f)
        threshold = metadata['threshold']
    
    # Compute reconstruction error
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        scores = errors.cpu().numpy()
    
    metrics = compute_reconstruction_metrics(y_test, scores, threshold)
    
    results = {
        'model': 'Centralized (Full Data)',
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'threshold': threshold,
        'y_pred': metrics['y_pred'],
        'scores': scores
    }
    
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"Threshold:    {threshold:.4f}")
    
    return results


def evaluate_federated(X_test, y_test):
    """Evaluate federated model (best HFL)"""
    print("\n" + "="*60)
    print("FEDERATED MODEL EVALUATION (HFL - federated_r40)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load best federated model
    fed_dir = 'federated_artifacts_r40'
    model = SimpleAutoencoder()
    checkpoint = torch.load(f'{fed_dir}/global_autoencoder.pt', map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load threshold
    with open(f'{fed_dir}/final_summary.json', 'r') as f:
        summary = json.load(f)
        threshold = summary.get('threshold', 0.03)  # fallback
    
    # Compute reconstruction error
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        scores = errors.cpu().numpy()
    
    metrics = compute_reconstruction_metrics(y_test, scores, threshold)
    
    results = {
        'model': 'Federated (HFL - r40)',
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'threshold': threshold,
        'y_pred': metrics['y_pred'],
        'scores': scores
    }
    
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"Threshold:    {threshold:.4f}")
    
    return results


def evaluate_vfl(X_test, y_test):
    """Evaluate VFL model"""
    print("\n" + "="*60)
    print("VFL MODEL EVALUATION (Semantic Grouping - 3 Clients)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load VFL models
    vfl_dir = 'vfl_artifacts_semantic_corrected'
    
    client_0_model = ClientEmbeddingModel(13, 32).to(device)
    client_0_model.load_state_dict(torch.load(f'{vfl_dir}/client_0_forward_metrics_model.pt', map_location=device))
    client_0_model.eval()
    
    client_1_model = ClientEmbeddingModel(11, 32).to(device)
    client_1_model.load_state_dict(torch.load(f'{vfl_dir}/client_1_backward_metrics_model.pt', map_location=device))
    client_1_model.eval()
    
    client_2_model = ClientEmbeddingModel(28, 32).to(device)
    client_2_model.load_state_dict(torch.load(f'{vfl_dir}/client_2_flow_and_flags_model.pt', map_location=device))
    client_2_model.eval()
    
    global_model = GlobalReconstructionModel(96, 52).to(device)
    global_model.load_state_dict(torch.load(f'{vfl_dir}/global_reconstruction_model.pt', map_location=device))
    global_model.eval()
    
    # Load threshold and feature partitioning
    with open(f'{vfl_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
        threshold = metadata.get('threshold', 0.03)
    
    with np.load(f'{vfl_dir}/scaler.npz', allow_pickle=True) as scaler_data:
        feature_names = [str(f) for f in scaler_data['features'].tolist()]
    feature_indices = get_feature_partition_indices(feature_names)
    
    # Inference
    scores = []
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        
        # Extract features for each client
        X_fwd = X_tensor[:, feature_indices['client_0_forward_metrics']]
        X_bwd = X_tensor[:, feature_indices['client_1_backward_metrics']]
        X_flow = X_tensor[:, feature_indices['client_2_flow_and_flags']]
        
        # Get embeddings
        emb_fwd = client_0_model(X_fwd)
        emb_bwd = client_1_model(X_bwd)
        emb_flow = client_2_model(X_flow)
        
        # Aggregate
        emb_agg = torch.cat([emb_fwd, emb_bwd, emb_flow], dim=1)
        
        # Reconstruct
        X_reconstructed = global_model(emb_agg)
        
        # Error
        errors = torch.mean((X_tensor - X_reconstructed) ** 2, dim=1)
        scores = errors.cpu().numpy()
    
    metrics = compute_reconstruction_metrics(y_test, scores, threshold)
    
    results = {
        'model': 'VFL (Split Learning - Semantic)',
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'threshold': threshold,
        'y_pred': metrics['y_pred'],
        'scores': scores
    }
    
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"Threshold:    {threshold:.4f}")
    
    return results


def evaluate_hybrid(hybrid_dir='hybrid_artifacts', hybrid_data_root='hybrid_data'):
    """Evaluate hybrid VHFL model across organizations and report weighted metrics."""
    print("\n" + "="*60)
    print("HYBRID MODEL EVALUATION (VHFL - VFL within org + FedAvg across orgs)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_path = os.path.join(hybrid_dir, 'final_summary.json')
    metadata_path = os.path.join(hybrid_dir, 'metadata.json')
    scaler_path = os.path.join(hybrid_dir, 'scaler.npz')

    with open(summary_path, 'r') as f:
        summary = json.load(f)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    threshold_quantile = float(summary.get('threshold_quantile', metadata.get('threshold_quantile', 0.99)))
    feature_sets = metadata.get('feature_sets', VFL_FEATURE_SETS)
    feature_indices_meta = metadata.get('feature_indices')
    embedding_dim = int(metadata.get('embedding_dim', 32))

    # Load scalers per organization
    with np.load(scaler_path, allow_pickle=True) as scaler_data:
        orgs = [str(o) for o in scaler_data['orgs'].tolist()]
        features = [str(f) for f in scaler_data['features'].tolist()]
        means = scaler_data['means']
        scales = scaler_data['scales']

    if feature_indices_meta is None:
        feature_indices = get_feature_partition_indices(features)
    else:
        feature_indices = {k: [int(i) for i in v] for k, v in feature_indices_meta.items()}

    # Load global hybrid models
    client_models = {}
    for client_name in feature_sets.keys():
        input_dim = len(feature_indices[client_name])
        model = ClientEmbeddingModel(input_dim, embedding_dim).to(device)
        model.load_state_dict(torch.load(os.path.join(hybrid_dir, f'{client_name}_model.pt'), map_location=device))
        model.eval()
        client_models[client_name] = model

    total_embedding_dim = embedding_dim * len(client_models)
    global_model = GlobalReconstructionModel(total_embedding_dim, len(features)).to(device)
    global_model.load_state_dict(torch.load(os.path.join(hybrid_dir, 'global_reconstruction_model.pt'), map_location=device))
    global_model.eval()

    client_order = list(feature_sets.keys())
    per_org_results = []

    for org_idx, org_name in enumerate(orgs):
        test_path = os.path.join(hybrid_data_root, org_name, 'test.csv')
        val_path = os.path.join(hybrid_data_root, org_name, 'val.csv')

        if not os.path.exists(test_path) or not os.path.exists(val_path):
            continue

        test_df = pd.read_csv(test_path)
        val_df = pd.read_csv(val_path)

        label_col = metadata.get('label_column', 'Attack Type')
        if label_col not in test_df.columns:
            label_col = 'Attack Type'

        y_test = (test_df[label_col].astype(str).str.strip().str.lower() != 'normal traffic').astype(int).values
        y_val = (val_df[label_col].astype(str).str.strip().str.lower() != 'normal traffic').astype(int).values

        x_test_df = test_df.drop(columns=[label_col]).copy()
        x_val_df = val_df.drop(columns=[label_col]).copy()

        x_test = x_test_df[features].to_numpy(dtype=np.float32)
        x_val = x_val_df[features].to_numpy(dtype=np.float32)

        mean = means[org_idx]
        scale = np.where(scales[org_idx] == 0, 1.0, scales[org_idx])
        x_test = ((x_test - mean) / scale).astype(np.float32)
        x_val = ((x_val - mean) / scale).astype(np.float32)

        def compute_scores(x_np):
            with torch.no_grad():
                x_full = torch.FloatTensor(x_np).to(device)
                embeddings = []
                for client_name in client_order:
                    idx = feature_indices[client_name]
                    x_part = torch.FloatTensor(x_np[:, idx]).to(device)
                    embeddings.append(client_models[client_name](x_part))
                agg = torch.cat(embeddings, dim=1)
                recon = global_model(agg)
                mse = torch.mean((x_full - recon) ** 2, dim=1)
                return mse.cpu().numpy()

        val_scores = compute_scores(x_val)
        val_normal_mask = y_val == 0
        threshold = float(np.quantile(val_scores[val_normal_mask], threshold_quantile)) if np.any(val_normal_mask) else float(np.quantile(val_scores, threshold_quantile))

        test_scores = compute_scores(x_test)
        metrics = compute_reconstruction_metrics(y_test, test_scores, threshold)
        metrics['threshold'] = threshold

        per_org_results.append({
            'org': org_name,
            'n_test': int(len(y_test)),
            'metrics': metrics,
        })

    if not per_org_results:
        raise RuntimeError("No organization test data found for hybrid evaluation.")

    weights = np.array([r['n_test'] for r in per_org_results], dtype=np.float64)
    weight_sum = float(np.sum(weights))

    def wavg(metric_name):
        vals = np.array([r['metrics'][metric_name] for r in per_org_results], dtype=np.float64)
        return float(np.sum(vals * weights) / weight_sum)

    results = {
        'model': 'VHFL (Vertical-Horizontal Hybrid)',
        'f1': wavg('f1'),
        'precision': wavg('precision'),
        'recall': wavg('recall'),
        'roc_auc': wavg('roc_auc'),
        'pr_auc': wavg('pr_auc'),
        'threshold': wavg('threshold'),
        'y_pred': None,
        'scores': None,
    }

    print(f"Organizations evaluated: {len(per_org_results)}")
    print(f"F1 Score:     {results['f1']:.4f}")
    print(f"Precision:    {results['precision']:.4f}")
    print(f"Recall:       {results['recall']:.4f}")
    print(f"ROC-AUC:      {results['roc_auc']:.4f}")
    print(f"PR-AUC:       {results['pr_auc']:.4f}")
    print(f"Threshold(q): {results['threshold']:.4f} (weighted avg)")

    return results


def compare_models(results_list):
    """Create comparison summary"""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    df = pd.DataFrame([
        {
            'Model': r['model'],
            'F1': f"{r['f1']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'ROC-AUC': f"{r['roc_auc']:.4f}",
            'PR-AUC': f"{r['pr_auc']:.4f}",
        }
        for r in results_list
    ])
    
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('comparison_vfl_vs_hfl.csv', index=False)
    print("\n[+] Saved to: comparison_vfl_vs_hfl.csv")
    
    # Detailed comparison
    print("\nDETAILED ANALYSIS:")
    print("-" * 60)
    
    for i, r in enumerate(results_list):
        model_name = r['model']
        f1 = r['f1']
        roc_auc = r['roc_auc']
        
        print(f"\n{i+1}. {model_name}")
        
        if f1 > 0.6:
            print(f"   [+] Strong F1 score ({f1:.4f})")
        elif f1 > 0.4:
            print(f"   [!] Moderate F1 score ({f1:.4f})")
        else:
            print(f"   [-] Low F1 score ({f1:.4f}) - needs tuning")
        
        if roc_auc > 0.85:
            print(f"   [+] Excellent ROC-AUC ({roc_auc:.4f})")
        elif roc_auc > 0.75:
            print(f"   [+] Good ROC-AUC ({roc_auc:.4f})")
        else:
            print(f"   [!] Fair ROC-AUC ({roc_auc:.4f})")
    
    # Comparison with best
    best_idx = np.argmax([r['f1'] for r in results_list])
    best_model = results_list[best_idx]['model']
    best_f1 = results_list[best_idx]['f1']
    
    print(f"\nBest Model (by F1): {best_model} ({best_f1:.4f})")
    
    # VFL vs others (if VFL present)
    vfl_results = [r for r in results_list if 'VFL' in r['model']]
    hfl_results = [r for r in results_list if 'Federated' in r['model']]
    
    if vfl_results and hfl_results:
        vfl_f1 = vfl_results[0]['f1']
        vfl_roc = vfl_results[0]['roc_auc']
        hfl_f1 = hfl_results[0]['f1']
        
        f1_diff = ((vfl_f1 - hfl_f1) / hfl_f1) * 100
        print(f"\nVFL vs HFL:")
        print(f"  F1 Difference:  {f1_diff:.1f}% ({'worse' if f1_diff < 0 else 'better'})")
        print(f"  VFL ROC-AUC:    {vfl_roc:.4f} (competitive)")
        print(f"\n  Trade-off: VFL trades F1 for privacy/feature isolation")
        print(f"  ROC-AUC good for ranking/prioritization tasks")


def main():
    parser = argparse.ArgumentParser(description='Compare VFL vs HFL vs Centralized vs VHFL')
    parser.add_argument('--test-only', action='store_true', help='Only evaluate given model')
    parser.add_argument('--model', choices=['centralized', 'federated', 'vfl', 'hybrid'], help='Model to test')
    parser.add_argument('--hybrid-dir', type=str, default='hybrid_artifacts', help='Hybrid artifacts directory')
    parser.add_argument('--hybrid-data-root', type=str, default='hybrid_data', help='Hybrid partitioned data root')
    args = parser.parse_args()
    
    # Load data
    X_test, y_test = load_data()
    
    results = []
    
    if args.test_only and args.model == 'centralized':
        results.append(evaluate_centralized(X_test, y_test))
    elif args.test_only and args.model == 'federated':
        results.append(evaluate_federated(X_test, y_test))
    elif args.test_only and args.model == 'vfl':
        results.append(evaluate_vfl(X_test, y_test))
    elif args.test_only and args.model == 'hybrid':
        results.append(evaluate_hybrid(args.hybrid_dir, args.hybrid_data_root))
    else:
        # Evaluate all
        try:
            results.append(evaluate_centralized(X_test, y_test))
        except Exception as e:
            print(f"Error evaluating centralized: {e}")
        
        try:
            results.append(evaluate_federated(X_test, y_test))
        except Exception as e:
            print(f"Error evaluating federated: {e}")
        
        try:
            results.append(evaluate_vfl(X_test, y_test))
        except Exception as e:
            print(f"Error evaluating VFL: {e}")

        try:
            results.append(evaluate_hybrid(args.hybrid_dir, args.hybrid_data_root))
        except Exception as e:
            print(f"Error evaluating Hybrid VHFL: {e}")
    
    # Compare
    if results:
        compare_models(results)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
