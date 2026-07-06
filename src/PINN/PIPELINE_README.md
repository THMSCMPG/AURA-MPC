"""PINN Training Pipeline - NEW Architecture

This document describes the rebuilt PINN training pipeline for AURA-MPC.
Built on RK4TRAN synthetic data with sandbox RL validation and optional fine-tuning.
"""

# PINN Training Pipeline - Phase 1 Complete ✅

## Overview

The PINN training pipeline has been completely rebuilt to:
1. **Pre-train** on RK4TRAN synthetic data (62k samples, 17 features, MC UQ)
2. **Validate** in sandbox RL environment with live visualization
3. **Fine-tune** based on RL insights (optional)

## New Architecture

### Directory Structure
```
src/PINN/
├── data/                    # Data pipeline
│   ├── __init__.py
│   ├── loaders.py          # RK4TRANDataset, create_dataloaders
│   ├── processors.py       # NumericNormalizer, DataProcessor
│   ├── uncertainty.py      # MC UQ handling
│   └── validate.py         # Real datasets (stubbed)
│
├── models/                  # Neural architectures
│   ├── __init__.py
│   └── pinn.py             # PINNSurrogate, PINNEnsemble, ResidualBlock
│
├── training/                # Training loops
│   ├── __init__.py
│   └── pretrain.py         # Trainer, compute_loss, weighted_mse_loss
│
├── sandbox/                 # RL + Validation
│   ├── __init__.py
│   ├── viewers/            # 3D visualization (to implement)
│   ├── environment.py      # Panel environment (to update)
│   ├── training.py         # RL loop (to update)
│   └── integration.py      # PINN-RK4TRAN integration (to implement)
│
├── configs/                 # Configuration files
│   ├── pretrain.yaml       # Pre-training config
│   ├── sandbox.yaml        # RL environment config
│   └── finetune.yaml       # Fine-tuning config
│
├── scripts/                 # Entry points
│   ├── run_pretrain.py     # Pre-training CLI
│   ├── run_sandbox.py      # Sandbox RL CLI (placeholder)
│   ├── run_finetune.py     # Fine-tuning CLI (placeholder)
│   └── test.py             # Validation test suite
│
└── reference/              # DEPRECATED - reference only
    └── [old code]
```

## Components

### Data Module (`data/`)

**RK4TRANDataset** - PyTorch Dataset for CSV files
- Loads RK4TRAN output CSVs (spacious, comfortable, cramped, lazy grids)
- Handles 17 fields: location, time, weather (7), panel state (4), outputs (4)
- Stores MC uncertainty bounds (sigma fields)
- Supports optional normalization

**NumericNormalizer** - Min-max field normalization
- Saves/loads normalization parameters
- Supports per-field normalization
- Ensures reproducibility

**UncertaintyProcessor** - MC UQ handling
- Strategy: weighted (use sigma for loss weighting)
- Accumulates UQ statistics across batches
- Provides loss weights based on uncertainty bounds

**DataLoader Factory** - train/val/test split
- 80/10/10 split with configurable seed
- Returns PyTorch DataLoaders with batching

### Model Module (`models/`)

**PINNSurrogate** - Main PINN architecture
- Input: Weather (T_amb, wind, humidity, irradiance, clouds, pressure) + panel state (height, pitch, roll, yaw) + location (lat, lon, elevation) + time
- Hidden: 4 residual blocks with batch norm and dropout
- Output: T_operating, T_operating_sigma, eta, eta_sigma
- ~340k parameters (configurable)

**ResidualBlock** - Skip connection with batch norm
- Helps with training stability and gradient flow

**PINNEnsemble** - Ensemble for improved UQ (research feature)
- Multiple PINN instances
- Computes ensemble mean and std

### Training Module (`training/`)

**Trainer** - Pre-training orchestrator
- Supervised learning on RK4TRAN synthetic data
- Adam optimizer + cosine annealing LR scheduler
- Weighted MSE loss with UQ weighting
- Best model checkpointing
- CSV logging and TensorBoard support

**Loss Functions**
- `weighted_mse_loss`: MSE with optional sample weighting
- `compute_loss`: Combines T and eta losses

## Usage

### 1. Pre-training

```bash
cd src/PINN

# Basic pre-training
python scripts/run_pretrain.py --config configs/pretrain.yaml

# Custom settings
python scripts/run_pretrain.py \
  --config configs/pretrain.yaml \
  --csv-dir ../RK4TRAN/work/ \
  --epochs 150 \
  --batch-size 32 \
  --device cuda
```

**Output**:
- `outputs/pretrain/checkpoints/best_model.pt` - Best model checkpoint
- `outputs/pretrain/logs/metrics.csv` - Training metrics
- `outputs/pretrain/logs/pretrain/` - TensorBoard logs

### 2. Sandbox RL (To Implement)

```bash
python scripts/run_sandbox.py --config configs/sandbox.yaml
```

Will include:
- Load pre-trained PINN checkpoint
- Live 3D panel visualization
- Compare PINN vs RK4TRAN predictions
- Train control policy via RL (PPO/A2C)

### 3. Fine-tuning (To Implement)

```bash
python scripts/run_finetune.py \
  --config configs/finetune.yaml \
  --checkpoint outputs/pretrain/checkpoints/best_model.pt
```

## Configuration Files

### pretrain.yaml
- Data paths (csv_data_dir, output directories)
- Data split and batch size
- Model architecture (input_dim=18, hidden_dim=128, num_residual_blocks=4)
- Training hyperparameters (epochs=100, lr=1e-3)
- Scheduler config (cosine annealing)

### sandbox.yaml
- PINN checkpoint path
- Physics constants (beta_Pmax, eta_ref, T_ref_K)
- RL parameters (episodes, discount_gamma, learning_rate)
- Reward weights (capture vs temperature control)
- Environment bounds (lat/lon, temperature, wind, irradiance)
- Fortran binary path for RK4TRAN integration
- Viewer settings

### finetune.yaml
- Pre-trained checkpoint path
- RL experience data path (for combined training)
- Lower learning rate (1e-4)
- Fewer epochs (20 for refinement)

## Data Flow

```
RK4TRAN CSV Files (62k samples)
    ↓
RK4TRANDataset
    ↓
create_dataloaders() → (train_loader, val_loader, test_loader)
    ↓
Trainer.train()
    ├─ Forward pass: x → model → {T_operating, T_sigma, eta, eta_sigma}
    ├─ Loss: weighted_mse_loss (sigma-weighted)
    ├─ Backward: gradient descent
    └─ Checkpointing: save best model
    ↓
outputs/pretrain/checkpoints/best_model.pt
```

## CSV Input Format

RK4TRAN output CSV with these columns:
```
location,time,T_amb,wind_speed,wind_dir,humidity,irradiance,cloud_cover,pressure,
pv_height,pitch,roll,yaw,T_operating,T_operating_sigma,eta,eta_sigma
```

Example row:
```
151.21 -33.87 39.0, 0 6 105 4 2024, 295.5, 10.06, 272.9, 0.021, 1026.3, 0.88, 70363.9,
2.1, -40.5, -49.3, -137.4, 0.0021, 0.067, -0.0000, 0.00005
```

## Key Features

✅ **Done**
- CSV data loading with uncertainty handling
- Min-max normalization with save/load
- PyTorch Dataset and DataLoader creation
- PINN model with residual blocks
- Pre-training loop with early stopping
- Weighted loss for MC UQ
- Config-driven training
- Comprehensive error handling

🔄 **In Progress**
- Sandbox RL integration
- 3D live viewer
- PINN-RK4TRAN comparison

❌ **Not Yet**
- Physics-informed loss terms
- Ensemble training loop
- Real dataset integration (Sandia, spectra)

## Testing

Run the test suite:
```bash
python scripts/test.py
```

Tests:
- Model instantiation and parameter count
- Forward pass with random input
- Data loading from CSVs (if available)
- DataLoader batching
- Output shapes

## Dependencies

- PyTorch >= 1.10
- NumPy
- PyYAML
- TensorBoard

Install with:
```bash
pip install torch numpy pyyaml tensorboard
```

## Architecture Decisions

1. **Model**: 4 residual blocks with batch norm
   - Reason: Good balance between expressiveness and training stability
   
2. **Loss**: Weighted MSE with sigma-weighting
   - Reason: Incorporates MC uncertainty bounds naturally
   
3. **Scheduler**: Cosine annealing
   - Reason: Stable convergence, no manual rate reduction needed
   
4. **Normalization**: Per-field min-max
   - Reason: Simple, deterministic, saves easily
   
5. **Data split**: 80/10/10 with seed
   - Reason: Standard, reproducible across runs

## Known Limitations

- Physics loss not yet implemented (optional research feature)
- Ensemble not integrated into training loop
- Sandbox RL only partially implemented
- Real dataset loaders stubbed (Sandia, spectra)
- No distributed training support
- Inference only on single samples (no batch optimization)

## Next Steps

1. **Test with real data**: Copy RK4TRAN CSV files and run full training
2. **Implement sandbox**: Integrate PINN with RK4TRAN binary + RL training
3. **Build viewer**: 3D visualization of panel, weather, predictions
4. **Optimize**: Profile training, tune hyperparameters, add callbacks
5. **Validate**: Test on held-out data and real datasets (future)

## Debugging

### Common Issues

**"No CSV files found"**
- Check csv_data_dir in config points to RK4TRAN output

**"Model doesn't converge"**
- Try lower learning rate (1e-4)
- Increase batch size (128)
- Check input normalization

**"Out of memory"**
- Reduce batch_size in config
- Reduce model hidden_dim
- Use num_workers=0 in DataLoader

**"CUDA out of memory"**
- Use device='cpu' to debug
- Reduce hidden_dim (128→64)
- Reduce batch_size (64→32)

## References

- Original PINN reference code: `reference/pinn/`
- RK4TRAN Fortran simulator: `../RK4TRAN/`
- Config reference: `configs/`

## Author Notes

Built as complete redesign from outdated reference code. Focuses on:
- Clean separation of concerns (data, models, training)
- Reproducibility (seeds, normalized configs)
- Scalability (DataLoaders, checkpoint management)
- Extensibility (easy to add new data sources, models, phases)

Intentionally NOT implemented:
- Real dataset loading (stubbed for future when data ready)
- Physics loss (can be added in post-training research phase)
- Distributed training (not needed for current scale)
