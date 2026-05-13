# cps_hybrid_fed_learning

Federated, vertical federated, and hybrid VHFL experiments for CICIDS2017 intrusion detection.

This workspace contains the training, partitioning, evaluation, and comparison scripts used to build
centralized, federated, VFL, and hybrid models on the cleaned CICIDS2017 dataset.

## Requirements

Install the Python dependencies listed in [requirements.txt](requirements.txt):

- numpy
- pandas
- scikit-learn
- torch

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The scripts expect a cleaned dataset named [cicids2017_cleaned.csv](cicids2017_cleaned.csv) in the
repository root.

## Main Workflows

### 1. Train the centralized autoencoder baseline

```powershell
python train_autoencoder.py --csv cicids2017_cleaned.csv
```

Outputs are written to `artifacts/` by default.

### 2. Create federated or hybrid partitions

```powershell
python partition_cicids.py --csv cicids2017_cleaned.csv --output-dir federated_data
python partition_hybrid_cicids.py --csv cicids2017_cleaned.csv --output-dir hybrid_data
```

### 3. Create VFL feature partitions

```powershell
python vfl_feature_partitioning.py
```

This generates VFL-ready datasets under `vfl_data/`.

### 4. Train the learning variants

```powershell
python vfl_train.py --csv cicids2017_cleaned.csv
python vfl_train_corrected.py --csv cicids2017_cleaned.csv
python federated_train.py --data-root federated_data_stratified
python hybrid_vhfl_train.py --data-root hybrid_data
```

The default output directories are:

- `vfl_artifacts/` or `vfl_artifacts_semantic/`
- `federated_artifacts/`
- `hybrid_artifacts/`

### 5. Tune the VFL threshold

```powershell
python vfl_threshold_tuning.py --vfl-dir vfl_artifacts_semantic_corrected
```

Results are written to `vfl_threshold_tuning/`.

### 6. Compare models and generate plots

```powershell
python evaluate_all_models.py
python compare_vfl_hfl.py
python create_comparison_visualizations.py
python create_vfl_weighted_metrics.py
```

## Script Guide

- [train_autoencoder.py](train_autoencoder.py): trains the centralized baseline and saves the model, scaler, and metadata.
- [partition_cicids.py](partition_cicids.py): creates federated partitions from the cleaned CSV.
- [partition_hybrid_cicids.py](partition_hybrid_cicids.py): creates hybrid organization partitions.
- [vfl_feature_partitioning.py](vfl_feature_partitioning.py): builds VFL feature splits.
- [vfl_train.py](vfl_train.py): trains the original VFL pipeline.
- [vfl_train_corrected.py](vfl_train_corrected.py): trains the corrected VFL pipeline.
- [federated_train.py](federated_train.py): trains the federated baseline.
- [hybrid_vhfl_train.py](hybrid_vhfl_train.py): trains the hybrid VHFL model.
- [vfl_threshold_tuning.py](vfl_threshold_tuning.py): searches for the best VFL threshold.
- [evaluate_all_models.py](evaluate_all_models.py): builds a unified comparison across models.
- [compare_vfl_hfl.py](compare_vfl_hfl.py): compares VFL, HFL, centralized, and hybrid results.

## Notes

- Generated folders are recreated by the scripts when needed.
- If you remove outputs, rerun the relevant script to regenerate them.
- The cleaned dataset is the main input; most other files are derived artifacts.
