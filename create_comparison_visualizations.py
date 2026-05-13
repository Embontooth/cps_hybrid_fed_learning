#!/usr/bin/env python3
"""
Create comprehensive performance comparison visualization

Compares:
1. VFL original (threshold=0.0300, F1=0.2598)  
2. VFL tuned (threshold=4.1892, F1=0.6342)
3. HFL baseline (F1=0.3861)
4. Centralized baseline (F1=0.3780)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Rectangle

def create_comprehensive_comparison():
    """Create comprehensive before/after comparison"""
    
    # Data
    models = ['Original\n(q=0.99)', 'Tuned\n(q=0.86)', 'HFL\nBaseline', 'Centralized\nBaseline']
    f1_scores = [0.2598, 0.6342, 0.3861, 0.3780]
    precisions = [0.1558, 0.6996, 0.2618, 0.2682]
    recalls = [0.7812, 0.5800, 0.7348, 0.6397]
    roc_aucs = [0.6757, 0.6757, 0.7534, 0.7714]
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. F1 Score Comparison
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#ff6b6b', '#51cf66', '#4ecdc4', '#4586d1']
    bars1 = ax1.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Highlight improvement
    ax1.annotate('', xy=(1, f1_scores[1]), xytext=(0, f1_scores[0]),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
    ax1.text(0.5, (f1_scores[0] + f1_scores[1])/2 + 0.05, 
            f'+143%', fontsize=12, fontweight='bold', color='green', ha='center')
    
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Score Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 0.85])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Precision-Recall Tradeoff
    ax2 = plt.subplot(2, 3, 2)
    x_pos = np.arange(len(models))
    width = 0.35
    
    bars_prec = ax2.bar(x_pos - width/2, precisions, width, label='Precision', 
                        color='#ff9f43', alpha=0.8, edgecolor='black', linewidth=1)
    bars_rec = ax2.bar(x_pos + width/2, recalls, width, label='Recall',
                       color='#a29bfe', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Precision vs Recall', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models)
    ax2.set_ylim([0, 1.0])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars_prec, bars_rec]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. ROC-AUC Scores
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(models, roc_aucs, color=['#ffa502', '#51cf66', '#4ecdc4', '#4586d1'], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
    ax3.set_title('ROC-AUC Score', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    # Add value labels
    for bar, val in zip(bars3, roc_aucs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Threshold Tuning Progress
    ax4 = plt.subplot(2, 3, 4)
    quantiles = np.array([0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
                          0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98])
    f1_tuning = np.array([0.3689, 0.3797, 0.3927, 0.4074, 0.4237, 0.4411, 0.4602, 0.4803,
                         0.5030, 0.5264, 0.5516, 0.5760, 0.6076, 0.6342, 0.6271, 0.5651,
                         0.5037, 0.3948, 0.2853, 0.1420])
    
    ax4.plot(quantiles, f1_tuning, 'o-', linewidth=2.5, markersize=8, color='#2196F3', label='F1 during tuning')
    ax4.axvline(0.86, color='green', linestyle='--', linewidth=2, label='Optimal (q=0.86)')
    ax4.scatter([0.86], [0.6342], color='red', s=300, marker='*', zorder=5, label='Best Found')
    
    ax4.set_xlabel('Threshold Quantile', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax4.set_title('Threshold Tuning Progress', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_ylim([0.1, 0.7])
    
    # 5. Performance Improvement Summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    summary_text = f"""
THRESHOLD TUNING RESULTS
{'='*50}

Original VFL (q=0.99):
  F1:        0.2598
  Precision: 0.1558
  Recall:    0.7812
  
Tuned VFL (q=0.86):
  F1:        0.6342
  Precision: 0.6996
  Recall:    0.5800

Improvements:
  F1 Score:   +144.1% (0.2598 -> 0.6342)
  Precision:  +349.0% (0.1558 -> 0.6996)
  Recall:     -25.7%  (0.7812 -> 0.5800)
  
Status: Better precision/recall balance
        F1 nearly matches HFL baseline
"""
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 6. Comparison with Baselines
    ax6 = plt.subplot(2, 3, 6)
    
    comparison_data = {
        'Original': {'x': 0.2598, 'color': '#ff6b6b', 'label': 'VFL Original'},
        'Tuned': {'x': 0.6342, 'color': '#51cf66', 'label': 'VFL Tuned'},
        'HFL': {'x': 0.3861, 'color': '#4ecdc4', 'label': 'HFL Baseline'},
        'Centralized': {'x': 0.3780, 'color': '#4586d1', 'label': 'Centralized'}
    }
    
    # Horizontal bar chart
    y_pos = np.arange(len(comparison_data))
    values = [comparison_data[k]['x'] for k in comparison_data.keys()]
    colors_bars = [comparison_data[k]['color'] for k in comparison_data.keys()]
    labels = [comparison_data[k]['label'] for k in comparison_data.keys()]
    
    bars6 = ax6.barh(y_pos, values, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([comparison_data[k]['label'] for k in comparison_data.keys()], fontsize=11)
    ax6.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax6.set_title('F1 Score Ranking', fontsize=13, fontweight='bold')
    ax6.set_xlim([0, 0.75])
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars6, values):
        width = bar.get_width()
        ax6.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('VFL Threshold Tuning: Complete Performance Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('vfl_threshold_tuning/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("[+] Saved: vfl_threshold_tuning/comprehensive_comparison.png")
    plt.close()


def create_metrics_table():
    """Create detailed metrics table"""
    
    data = {
        'Model': ['VFL Original', 'VFL Tuned', 'HFL Baseline', 'Centralized'],
        'Threshold': [0.0300, 4.1892, 0.0300, 0.0713],
        'Quantile': [0.99, 0.86, 0.99, 0.99],
        'F1 Score': [0.2598, 0.6342, 0.3861, 0.3780],
        'Precision': [0.1558, 0.6996, 0.2618, 0.2682],
        'Recall': [0.7812, 0.5800, 0.7348, 0.6397],
        'ROC-AUC': [0.6757, 0.6757, 0.7534, 0.7714],
        'Privacy': ['HIGH', 'HIGH', 'MEDIUM', 'LOW']
    }
    
    df = pd.DataFrame(data)
    df.to_csv('vfl_threshold_tuning/metrics_comparison.csv', index=False)
    print("[+] Saved: vfl_threshold_tuning/metrics_comparison.csv")
    
    # Display
    print("\n" + "="*100)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)


def main():
    print("Creating comprehensive comparison visualizations...")
    create_comprehensive_comparison()
    create_metrics_table()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files in vfl_threshold_tuning/:")
    print("  1. comprehensive_comparison.png - Main comparison dashboard")
    print("  2. vfl_threshold_tuning.png - Metrics vs quantile")
    print("  3. vfl_precision_recall_curve.png - PR curve")
    print("  4. vfl_confusion_matrix.png - Confusion matrix")
    print("  5. metrics_comparison.csv - Detailed metrics table")
    print("  6. threshold_tuning_results.csv - All tuning results")
    print("  7. best_threshold.json - Optimal threshold config")


if __name__ == '__main__':
    main()
