# Training Curves

`scripts/plot_training.py` visualizes the training history saved by `train.py` in `training_history.json`.

## Usage

```bash
# Default (reads checkpoints/training_history.json)
python scripts/plot_training.py

# Custom paths
python scripts/plot_training.py --history checkpoints/training_history.json --outdir outputs/plots
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--history` | `checkpoints/training_history.json` | Path to history JSON |
| `--outdir` | `outputs/plots` | Output directory for the PNG |

## Output

A single PNG (`training_curves.png`) with six panels (four if metrics are not yet available):

| Panel | Content |
|---|---|
| **Total Loss** | Train + val total loss per epoch |
| **Species Loss** | Train + val species loss |
| **Environmental Loss** | Train + val env MSE |
| **Learning Rate** | LR schedule (log scale) |
| **Validation mAP** | Mean average precision (if recorded) |
| **Top-k Recall** | Top-10 and top-30 recall (if recorded) |

!!! note
    Metric panels (mAP, top-k recall) are only shown when the training history contains `val_map` and `val_top10_recall` keys. These are recorded automatically by `train.py` starting with the current version.
