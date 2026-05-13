#!/usr/bin/env python3
"""
VFL Threshold Tuning - Find optimal threshold for anomaly detection

This script tests different threshold values (quantiles) and evaluates
performance metrics (F1, precision, recall, ROC-AUC) to find the optimal threshold.
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
import matplotlib.pyplot as plt
import argparse


class ClientEmbeddingModel(nn.Module):
    """Local model at each VFL client"""
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
    """Server model for VFL"""
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


def load_test_data():
    """Load test data"""
    print("Loading test data...")
    
    test_df = pd.read_csv('federated_data_stratified/client_00/test.csv')
    X_test = test_df.drop(columns=['Attack Type']).values.astype(np.float32)
    y_test = (test_df['Attack Type'] != 'Normal Traffic').astype(int).values
    
    with np.load('artifacts/scaler.npz', allow_pickle=True) as data:
        mean = data['mean']
        scale = data['scale']
    
    X_test = (X_test - mean) / (scale + 1e-8)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Labels: Normal={sum(y_test==0)}, Attack={sum(y_test==1)}")
    
    return X_test, y_test


def compute_vfl_scores(X_test, vfl_dir='vfl_artifacts_semantic_corrected'):
    """Compute reconstruction error scores for VFL"""
    print("\nComputing VFL reconstruction errors...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
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
    
    # Feature indices
    forward_indices = list(range(0, 13))
    backward_indices = list(range(13, 24))
    flow_indices = list(range(24, 52))
    
    # Compute scores
    scores = []
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        
        X_fwd = X_tensor[:, forward_indices]
        X_bwd = X_tensor[:, backward_indices]
        X_flow = X_tensor[:, flow_indices]
        
        emb_fwd = client_0_model(X_fwd)
        emb_bwd = client_1_model(X_bwd)
        emb_flow = client_2_model(X_flow)
        
        emb_agg = torch.cat([emb_fwd, emb_bwd, emb_flow], dim=1)
        X_reconstructed = global_model(emb_agg)
        
        errors = torch.mean((X_tensor - X_reconstructed) ** 2, dim=1)
        scores = errors.cpu().numpy()
    
    return scores


def tune_threshold(y_test, scores, quantiles=None):
    """Test different thresholds and return metrics"""
    
    if quantiles is None:
        quantiles = np.arange(0.60, 1.00, 0.02)  # Test 0.60 to 0.98
    
    print(f"\nTesting {len(quantiles)} threshold values...")
    print("-" * 90)
    print(f"{'Quantile':<12}{'Threshold':<12}{'F1':<10}{'Precision':<12}{'Recall':<10}{'ROC-AUC':<10}")
    print("-" * 90)
    
    results = []
    
    for quantile in quantiles:
        threshold = np.quantile(scores, quantile)
        y_pred = (scores > threshold).astype(int)
        
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, scores)
        
        results.append({
            'quantile': quantile,
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'y_pred': y_pred
        })
        
        print(f"{quantile:<12.2f}{threshold:<12.6f}{f1:<10.4f}{precision:<12.4f}{recall:<10.4f}{roc_auc:<10.4f}")
    
    print("-" * 90)
    
    # Find best by F1
    best_result = max(results, key=lambda x: x['f1'])
    
    print(f"\n[BEST by F1]: Quantile {best_result['quantile']:.2f}")
    print(f"  Threshold: {best_result['threshold']:.6f}")
    print(f"  F1:        {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  ROC-AUC:   {best_result['roc_auc']:.4f}")
    
    return results, best_result


def create_visualizations(y_test, results, best_result, output_dir='vfl_threshold_tuning'):
    """Create performance visualization graphs"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    quantiles = [r['quantile'] for r in results]
    f1_scores = [r['f1'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    
    # Figure 1: F1, Precision, Recall vs Quantile
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(quantiles, f1_scores, 'o-', label='F1 Score', linewidth=2, markersize=6)
    ax.plot(quantiles, precisions, 's-', label='Precision', linewidth=2, markersize=6)
    ax.plot(quantiles, recalls, '^-', label='Recall', linewidth=2, markersize=6)
    
    # Mark best threshold
    best_idx = quantiles.index(best_result['quantile'])
    ax.axvline(best_result['quantile'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f"Best Threshold (q={best_result['quantile']:.2f})")
    ax.scatter([best_result['quantile']], [best_result['f1']], color='red', s=200, marker='*', zorder=5)
    
    ax.set_xlabel('Threshold Quantile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('VFL Threshold Tuning: Performance Metrics vs Quantile', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vfl_threshold_tuning.png', dpi=300, bbox_inches='tight')
    print(f"\n[+] Saved: {output_dir}/vfl_threshold_tuning.png")
    plt.close()
    
    # Figure 2: Precision-Recall Tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(recalls, precisions, 'o-', linewidth=2, markersize=8, color='blue')
    
    # Mark best threshold
    ax.scatter([best_result['recall']], [best_result['precision']], 
               color='red', s=300, marker='*', zorder=5, label=f"Best (q={best_result['quantile']:.2f})")
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('VFL Precision-Recall Tradeoff Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vfl_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"[+] Saved: {output_dir}/vfl_precision_recall_curve.png")
    plt.close()
    
    # Figure 3: Confusion Matrix for Best Threshold
    from sklearn.metrics import confusion_matrix
    y_pred = best_result['y_pred']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    # Labels and ticks
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Attack'], fontsize=12)
    ax.set_yticklabels(['Normal', 'Attack'], fontsize=12)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'VFL Confusion Matrix (Quantile={best_result["quantile"]:.2f})', 
                fontsize=13, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vfl_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"[+] Saved: {output_dir}/vfl_confusion_matrix.png")
    plt.close()
    
    # Figure 4: Comparison with original threshold
    original_q = 0.99
    original_result = next((r for r in results if abs(r['quantile'] - original_q) < 0.005), None)
    
    if original_result:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['F1', 'Precision', 'Recall']
        original = [original_result['f1'], original_result['precision'], original_result['recall']]
        tuned = [best_result['f1'], best_result['precision'], best_result['recall']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original, width, label=f"Original (q=0.99)", alpha=0.8)
        bars2 = ax.bar(x + width/2, tuned, width, label=f"Tuned (q={best_result['quantile']:.2f})", alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('VFL Performance: Original vs Tuned Threshold', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/vfl_improvement_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[+] Saved: {output_dir}/vfl_improvement_comparison.png")
        plt.close()


def save_results(results, best_result, output_dir='vfl_threshold_tuning'):
    """Save tuning results to CSV and JSON"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame([
        {
            'quantile': r['quantile'],
            'threshold': r['threshold'],
            'f1': r['f1'],
            'precision': r['precision'],
            'recall': r['recall'],
            'roc_auc': r['roc_auc']
        }
        for r in results
    ])
    
    df.to_csv(f'{output_dir}/threshold_tuning_results.csv', index=False)
    print(f"\n[+] Saved: {output_dir}/threshold_tuning_results.csv")
    
    # Save best result
    best_data = {
        'quantile': float(best_result['quantile']),
        'threshold': float(best_result['threshold']),
        'f1': float(best_result['f1']),
        'precision': float(best_result['precision']),
        'recall': float(best_result['recall']),
        'roc_auc': float(best_result['roc_auc']),
        'improvement_f1': float(best_result['f1'] - results[-1]['f1']) if len(results) > 0 else 0
    }
    
    with open(f'{output_dir}/best_threshold.json', 'w') as f:
        json.dump(best_data, f, indent=2)
    
    print(f"[+] Saved: {output_dir}/best_threshold.json")


def main():
    parser = argparse.ArgumentParser(description='VFL Threshold Tuning')
    parser.add_argument('--vfl-dir', default='vfl_artifacts_semantic_corrected', help='VFL model directory')
    parser.add_argument('--output-dir', default='vfl_threshold_tuning', help='Output directory for results')
    parser.add_argument('--quantile-step', type=float, default=0.02, help='Quantile step size')
    args = parser.parse_args()
    
    # Load data
    X_test, y_test = load_test_data()
    
    # Compute scores
    scores = compute_vfl_scores(X_test, args.vfl_dir)
    
    # Tune threshold
    quantiles = np.arange(0.60, 1.00, args.quantile_step)
    results, best_result = tune_threshold(y_test, scores, quantiles)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(y_test, results, best_result, args.output_dir)
    
    # Save results
    save_results(results, best_result, args.output_dir)
    
    print("\n" + "="*70)
    print("THRESHOLD TUNING COMPLETE!")
    print("="*70)
    print(f"\nOptimal Threshold Found:")
    print(f"  Quantile:   {best_result['quantile']:.2f}")
    print(f"  Threshold:  {best_result['threshold']:.6f}")
    print(f"  F1 Score:   {best_result['f1']:.4f}")
    print(f"  Precision:  {best_result['precision']:.4f}")
    print(f"  Recall:     {best_result['recall']:.4f}")
    print(f"\nResults saved to: {args.output_dir}/")
    print("  - threshold_tuning_results.csv")
    print("  - best_threshold.json")
    print("  - vfl_threshold_tuning.png (metrics vs quantile)")
    print("  - vfl_precision_recall_curve.png")
    print("  - vfl_confusion_matrix.png")
    print("  - vfl_improvement_comparison.png")


if __name__ == '__main__':
    main()
