"""
Vertical Federated Learning (VFL) Feature Partitioning Analysis for CICIDS2017

In VFL, instead of distributing DATA SAMPLES across clients (HFL),
we distribute FEATURES across clients. Each client has:
- ALL data samples
- A subset of features
- A local model that processes its features
- Aggregation happens at embedding/prediction level

Advantages:
- Privacy: Features stay within each "feature owner" (e.g., different departments)
- Use case: When different organizations own different feature sets
- Example: Network team has port/protocol features, Security team has flag features
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

# The 52 features in CICIDS2017
ALL_FEATURES = [
    # Network/Flow Level (4 features)
    "Destination Port",
    "Flow Duration",
    
    # Forward Packet Statistics (8 features)
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    
    # Backward Packet Statistics (8 features)
    "Total Bwd Packets",
    "Total Length of Bwd Packets",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    
    # Flow Rate Metrics (2 features)
    "Flow Bytes/s",
    "Flow Packets/s",
    
    # Flow Inter-Arrival Time (4 features)
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    
    # Forward IAT (5 features)
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    
    # Backward IAT (5 features)
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    
    # Header Lengths (2 features)
    "Fwd Header Length",
    "Bwd Header Length",
    
    # Per-packet Rates (2 features)
    "Fwd Packets/s",
    "Bwd Packets/s",
    
    # Packet Size Statistics (4 features)
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    
    # TCP Flags (3 features)
    "FIN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    
    # Additional Metrics (5 features)
    "Average Packet Size",
    "Subflow Fwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    
    # Timing (6 features)
    "Active Mean",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Max",
    "Idle Min",
]

# VFL PARTITIONING STRATEGIES
PARTITIONING_STRATEGIES = {
    "strategy_1_semantic": {
        "name": "Semantic Grouping (3 clients)",
        "description": "Group related features by semantic meaning",
        "clients": {
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
                "Total Bwd Packets",
                "Total Length of Bwd Packets",
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
    },
    "strategy_2_balanced_5clients": {
        "name": "Balanced 5-Client Partition",
        "description": "Distribute features evenly across 5 clients (~10 features each)",
        "clients": {
            "client_0_packet_counts": [
                "Destination Port",
                "Total Fwd Packets",
                "Total Bwd Packets",
                "Total Length of Fwd Packets",
                "Total Length of Bwd Packets",
                "FIN Flag Count",
                "PSH Flag Count",
                "ACK Flag Count",
                "Average Packet Size",
                "Subflow Fwd Bytes",
            ],
            "client_1_fwd_sizes": [
                "Fwd Packet Length Max",
                "Fwd Packet Length Min",
                "Fwd Packet Length Mean",
                "Fwd Packet Length Std",
                "Bwd Packet Length Max",
                "Bwd Packet Length Min",
                "Bwd Packet Length Mean",
                "Bwd Packet Length Std",
                "Min Packet Length",
                "Max Packet Length",
            ],
            "client_2_fwd_timing": [
                "Fwd IAT Total",
                "Fwd IAT Mean",
                "Fwd IAT Std",
                "Fwd IAT Max",
                "Fwd IAT Min",
                "Fwd Header Length",
                "Fwd Packets/s",
                "Flow Duration",
                "Flow Bytes/s",
                "Flow Packets/s",
            ],
            "client_3_bwd_timing": [
                "Bwd IAT Total",
                "Bwd IAT Mean",
                "Bwd IAT Std",
                "Bwd IAT Max",
                "Bwd IAT Min",
                "Bwd Header Length",
                "Bwd Packets/s",
                "Flow IAT Mean",
                "Flow IAT Std",
                "Flow IAT Max",
            ],
            "client_4_session_metrics": [
                "Flow IAT Min",
                "Packet Length Mean",
                "Packet Length Std",
                "Packet Length Variance",
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
    },
    "strategy_3_temporal_spatial": {
        "name": "Temporal vs Spatial Separation (2 clients)",
        "description": "Split based on temporal (timing) vs spatial (packet) characteristics",
        "clients": {
            "client_0_timing_stats": [
                "Flow Duration",
                "Flow IAT Mean",
                "Flow IAT Std",
                "Flow IAT Max",
                "Flow IAT Min",
                "Fwd IAT Total",
                "Fwd IAT Mean",
                "Fwd IAT Std",
                "Fwd IAT Max",
                "Fwd IAT Min",
                "Bwd IAT Total",
                "Bwd IAT Mean",
                "Bwd IAT Std",
                "Bwd IAT Max",
                "Bwd IAT Min",
                "Active Mean",
                "Active Max",
                "Active Min",
                "Idle Mean",
                "Idle Max",
                "Idle Min",
            ],
            "client_1_packet_spatial": [
                "Destination Port",
                "Total Fwd Packets",
                "Total Length of Fwd Packets",
                "Fwd Packet Length Max",
                "Fwd Packet Length Min",
                "Fwd Packet Length Mean",
                "Fwd Packet Length Std",
                "Bwd Packet Length Max",
                "Bwd Packet Length Min",
                "Bwd Packet Length Mean",
                "Bwd Packet Length Std",
                "Flow Bytes/s",
                "Flow Packets/s",
                "Fwd Header Length",
                "Bwd Header Length",
                "Fwd Packets/s",
                "Bwd Packets/s",
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
                "Total Bwd Packets",
                "Total Length of Bwd Packets",
            ],
        }
    },
}

def analyze_vfl_strategies():
    """Print analysis of all VFL strategies"""
    print("=" * 80)
    print("VERTICAL FEDERATED LEARNING (VFL) - FEATURE PARTITIONING STRATEGIES")
    print("=" * 80)
    print()
    
    for strategy_key, strategy_info in PARTITIONING_STRATEGIES.items():
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strategy_info['name']}")
        print(f"{'='*80}")
        print(f"Description: {strategy_info['description']}\n")
        
        total_features = 0
        for client_name, features in strategy_info['clients'].items():
            print(f"  {client_name}: {len(features)} features")
            total_features += len(features)
            for i, feat in enumerate(features, 1):
                print(f"    {i:2d}. {feat}")
            print()
        
        print(f"  Total features: {total_features}")
        print(f"  Coverage: {total_features}/{len(ALL_FEATURES)} ({100*total_features/len(ALL_FEATURES):.1f}%)")


def create_vfl_datasets(csv_path: Path, strategy_key: str):
    """Create VFL partitioned datasets based on selected strategy"""
    strategy = PARTITIONING_STRATEGIES[strategy_key]
    
    df = pd.read_csv(csv_path)
    label_col = "Attack Type"
    
    vfl_dir = Path("vfl_data") / strategy_key
    vfl_dir.mkdir(parents=True, exist_ok=True)
    
    # Save label separately
    labels = df[[label_col]].copy()
    labels.to_csv(vfl_dir / "labels.csv", index=False)
    
    # Save each client's features
    client_feature_map = {}
    for client_name, features in strategy['clients'].items():
        client_features_df = df[features].copy()
        
        # Handle missing or invalid features
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < len(features):
            missing = [f for f in features if f not in df.columns]
            print(f"Warning: {client_name} - Missing features: {missing}")
        
        client_features_df = df[available_features].copy()
        client_features_df.to_csv(vfl_dir / f"{client_name}_features.csv", index=False)
        client_feature_map[client_name] = available_features
    
    # Save metadata
    metadata = {
        "strategy": strategy_key,
        "strategy_name": strategy['name'],
        "description": strategy['description'],
        "total_samples": len(df),
        "total_features": len(ALL_FEATURES),
        "label_column": label_col,
        "clients": {
            client_name: {
                "n_features": len(features),
                "features": features
            }
            for client_name, features in strategy['clients'].items()
        }
    }
    
    with open(vfl_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Created VFL dataset for {strategy['name']}")
    print(f"  Location: {vfl_dir}")
    print(f"  Samples: {len(df)}")
    print(f"  Files created:")
    print(f"    - labels.csv")
    for client_name in strategy['clients'].keys():
        n_features = len(strategy['clients'][client_name])
        print(f"    - {client_name}_features.csv ({n_features} features)")
    print(f"    - metadata.json")


def print_feature_statistics(csv_path: Path):
    """Analyze feature statistics"""
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col != 'Attack Type']
    
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS FOR VFL PARTITIONING")
    print("=" * 80)
    print(f"{'Feature':<35} | {'Type':<8} | {'Mean':<12} | {'Type':<8}")
    print("-" * 80)
    
    for col in feature_cols:
        col_type = 'numeric'
        mean_val = df[col].mean()
        dtype = 'float' if isinstance(mean_val, float) else 'int'
        print(f"{col:<35} | {col_type:<8} | {mean_val:>12.2f} | {dtype:<8}")


def explain_vfl_architecture():
    """Explain the VFL architecture"""
    print("\n" + "=" * 80)
    print("VERTICAL FEDERATED LEARNING (VFL) ARCHITECTURE")
    print("=" * 80)
    print("""
HORIZONTAL FL (Current Implementation):
├─ Data Distribution: Samples distributed across clients
├─ Client 0: Rows 0-N/5 (all features)
├─ Client 1: Rows N/5-2N/5 (all features)
├─ Client 2: Rows 2N/5-3N/5 (all features)
├─ etc.
└─ Model: Each client trains locally, aggregates weights

VERTICAL FL (Proposed):
├─ Data Distribution: Features distributed across clients
├─ Client 0: All rows, subset of features (e.g., Forward metrics)
├─ Client 1: All rows, different subset (e.g., Backward metrics)
├─ Client 2: All rows, different subset (e.g., Flow/Flags)
├─ etc.
└─ Model: Split learning architecture with intermediate aggregation

KEY DIFFERENCES:
1. Data Alignment: VFL requires same sample IDs across clients
2. Model Structure: Each client processes its features independently
3. Aggregation: 
   - HFL: Aggregate weights after local training
   - VFL: Aggregate embeddings/predictions after feature processing
   
4. Privacy: 
   - HFL: Each client sees all features (privacy between sample owners)
   - VFL: Each client sees all samples but only their features (privacy between feature owners)

USE CASES FOR VFL:
• Multi-organization collaboration where organizations own different features
• Network monitoring: One org has network metrics, another has protocol metrics
• Healthcare: One org has patient demographics, another has test results
• Finance: One org has transaction features, another has account features
""")


def demonstrate_vfl_training_flow():
    """Show how VFL training would work"""
    print("\n" + "=" * 80)
    print("VFL TRAINING FLOW")
    print("=" * 80)
    print("""
Step 1: FEATURE PARTITIONING
  Forward_features (13 features) → Client A
  Backward_features (13 features) → Client B
  Flow_metrics (26 features) → Client C

Step 2: LOCAL EMBEDDING EXTRACTION
  Client A: Input (13 features) → Local NN → Embedding (32-dim)
  Client B: Input (13 features) → Local NN → Embedding (32-dim)
  Client C: Input (26 features) → Local NN → Embedding (32-dim)

Step 3: EMBEDDING AGGREGATION
  Aggregate embeddings: [32 + 32 + 32] = 96-dim combined embedding

Step 4: GLOBAL MODEL
  Combined embedding (96-dim) → Shared NN → Reconstruction (52 features)

Step 5: LOSS & BACKWARD PASS
  MSE Loss: ||X_reconstructed - X_original||²
  Gradients for local models sent back to each client
  Each client updates its local model

Step 6: FEDERATION
  Models trained for multiple rounds
  Models can remain private to each client
  Only embeddings cross the network (compressed, no raw features)

ADVANTAGES:
✓ Privacy: Each client keeps raw features private
✓ Scalability: Features distributed, computation parallelized
✓ Efficiency: Only embeddings (96-dim) transmitted, not raw data
✓ Flexible: Can add/remove clients by updating their feature set
""")


if __name__ == "__main__":
    csv_path = Path("cicids2017_cleaned.csv")
    
    # Print comprehensive analysis
    analyze_vfl_strategies()
    print_feature_statistics(csv_path)
    explain_vfl_architecture()
    demonstrate_vfl_training_flow()
    
    # Create VFL datasets
    print("\n" + "=" * 80)
    print("CREATING VFL DATASETS")
    print("=" * 80)
    
    create_vfl_datasets(csv_path, "strategy_1_semantic")
    create_vfl_datasets(csv_path, "strategy_2_balanced_5clients")
    create_vfl_datasets(csv_path, "strategy_3_temporal_spatial")
    
    print("\n✓ All VFL datasets created successfully!")
