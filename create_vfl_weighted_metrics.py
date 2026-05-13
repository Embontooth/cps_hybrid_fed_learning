"""
Create a weighted metrics comparison bar chart including VFL Tuned.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define the metrics for each model
models_data = {
    "VFL (Split Learning - Tuned)": {
        "precision": 0.6996,
        "recall": 0.5800,
        "f1": 0.6342,
        "pr_auc": 0.5046,
        "roc_auc": 0.6757,
    },
    "Federated (HFL - r40)": {
        "precision": 0.2618,
        "recall": 0.7348,
        "f1": 0.3861,
        "pr_auc": 0.5878,
        "roc_auc": 0.7534,
    },
    "Centralized (Full Data)": {
        "precision": 0.2682,
        "recall": 0.6397,
        "f1": 0.3780,
        "pr_auc": 0.6451,
        "roc_auc": 0.7714,
    },
}

# Metrics to plot
metrics_to_plot = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
metric_labels = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
}

# Sort by F1 score (descending)
sorted_models = sorted(models_data.items(), key=lambda x: x[1]["f1"], reverse=True)
model_names = [name for name, _ in sorted_models]
model_metrics = [metrics for _, metrics in sorted_models]

# Create the plot
x = np.arange(len(metrics_to_plot))
width = 0.8 / len(model_names)

fig, ax = plt.subplots(figsize=(12, 6))

# Define colors for each model
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, (model_name, metrics) in enumerate(sorted_models):
    vals = [metrics[m] for m in metrics_to_plot]
    ax.bar(x + i * width, vals, width=width, label=model_name, color=colors[i])

ax.set_title("Weighted Metric Comparison Across Models", fontsize=14, fontweight="bold")
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0.0, 1.0)
ax.set_xticks(x + (len(model_names) - 1) * width / 2)
ax.set_xticklabels([metric_labels[m] for m in metrics_to_plot], fontsize=11)
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()

# Save the figure
output_dir = Path("vfl_threshold_tuning")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "weighted_metrics_comparison_with_vfl.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"✅ Saved: {output_path}")

plt.close(fig)

# Print summary
print("\n📊 Weighted Metrics Comparison Summary:")
print("-" * 70)
for model_name, metrics in sorted_models:
    print(f"\n{model_name}:")
    for metric_key, label in metric_labels.items():
        print(f"  {label:10s}: {metrics[metric_key]:.4f}")
