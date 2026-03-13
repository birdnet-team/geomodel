# Model Export

Convert a trained checkpoint to portable inference formats for deployment.

## Quick Start

```bash
python convert.py                                    # ONNX FP16 (default)
python convert.py --formats onnx tflite_fp16         # multiple formats
python convert.py --formats all                      # everything
```

## Supported Formats

| Format | Flag | Size (medium) | Description |
|---|---|---|---|
| ONNX FP32 | `onnx` | ~19 MB | Full-precision ONNX model |
| ONNX FP16 | `onnx_fp16` | ~9 MB | Half-precision ONNX (default) |
| TFLite FP32 | `tflite` | ~19 MB | TensorFlow Lite, full precision |
| TFLite FP16 | `tflite_fp16` | ~10 MB | TensorFlow Lite, half precision |
| TFLite INT8 | `tflite_int8` | ~5 MB | TensorFlow Lite, dynamic-range quantisation |
| TF SavedModel | `tf` | ~19 MB | TensorFlow SavedModel directory |

Use `--formats all` to export everything at once.

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Source checkpoint |
| `--outdir` | `exports` | Output directory |
| `--formats` | `onnx_fp16` | Space-separated list of formats (or `all`) |
| `--tol` | `1e-4` | Base tolerance for numerical validation |
| `--device` | `auto` | Device for PyTorch reference model |
| `--fp16_io` | off | Convert model I/O to FP16 as well (see below) |

## FP16 I/O Behavior

By default, ONNX FP16 exports keep model **inputs and outputs in FP32** while
converting internal weights and activations to FP16.  This preserves full
coordinate precision (latitude, longitude, week) and keeps LayerNormalization
in FP32 for numerical stability.

Pass `--fp16_io` to convert I/O tensors to FP16 as well.  This is slightly
smaller but increases numerical divergence from the PyTorch reference.

| Mode | Max diff (typical) | Notes |
|---|---|---|
| FP16 weights, FP32 I/O (default) | ~0.013 | Best accuracy, recommended |
| Full FP16 (`--fp16_io`) | ~0.013 | Slightly lossy input precision |

## Export Wrapper

All exported models use a unified interface:

- **Input**: `(batch, 3)` float tensor — columns are `[latitude, longitude, week]`
- **Output**: `(batch, n_species)` float tensor — sigmoid probabilities

This differs from the raw PyTorch model which takes three separate tensors.  The wrapper applies sigmoid internally so outputs are directly interpretable as probabilities.

## Validation

After each conversion, the script runs automatic numerical validation:

1. A fixed set of 200 reference inputs (diverse lat/lon/week combinations) is generated
2. The PyTorch model produces reference outputs
3. The exported model is loaded back and run on the same inputs
4. Maximum and mean absolute differences are computed
5. If the max diff exceeds the tolerance, the conversion is marked as **FAIL**

Tolerances are format-aware:

| Format | Effective tolerance |
|---|---|
| ONNX FP32 | `tol` (1e-4) |
| ONNX FP16 (FP32 I/O, default) | 0.06 |
| ONNX FP16 (`--fp16_io`) | 0.05 |
| TFLite FP32 | `tol` (1e-4) |
| TFLite FP16 | `tol × 10` |
| TFLite INT8 | `tol × 100` |
| TF SavedModel | `tol` (1e-4) |

## Dependencies

ONNX formats require:

```bash
pip install onnx onnxruntime onnxscript onnxconverter-common
```

TFLite and TF SavedModel additionally require:

```bash
pip install tensorflow onnx2tf
```

The script will print a clear error message if a required package is missing — it does not fail silently.

## Output Structure

```
exports/
├── geomodel.onnx             # ONNX FP32
├── geomodel_fp16.onnx       # ONNX FP16
├── geomodel.tflite          # TFLite FP32
├── geomodel_fp16.tflite     # TFLite FP16
├── geomodel_int8.tflite     # TFLite INT8
├── saved_model/             # TF SavedModel
├── labels.txt               # Species vocabulary (copied from checkpoint dir)
└── MODEL_LICENSE.txt        # Model weights license (CC BY-SA 4.0)
```

## Running Exported Models

All exported formats share the same interface:

- **Input**: `(batch, 3)` float32 tensor — columns are `[latitude, longitude, week]`
- **Output**: `(batch, n_species)` float32 tensor — sigmoid probabilities

### Loading labels

```python
def load_labels(path="exports/labels.txt"):
    """Load species labels from labels.txt."""
    labels = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            labels.append({"code": parts[0], "sci": parts[1], "common": parts[2]})
    return labels
```

### PyTorch (.pt)

```python
import torch
import numpy as np
from model.model import BirdNETGeoModel

checkpoint = torch.load("checkpoints/checkpoint_best.pt", map_location="cpu",
                        weights_only=False)
cfg = checkpoint["model_config"]
model = BirdNETGeoModel(
    n_species=cfg["n_species"],
    n_env_features=cfg["n_env_features"],
    model_scale=cfg["model_scale"],
    coord_harmonics=cfg.get("coord_harmonics", 4),
    week_harmonics=cfg.get("week_harmonics", 8),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

lat = torch.tensor([52.5])
lon = torch.tensor([13.4])
week = torch.tensor([22.0])

with torch.no_grad():
    logits = model(lat, lon, week, return_env=False)["species_logits"]
    probs = torch.sigmoid(logits)          # (1, n_species)

labels = load_labels("checkpoints/labels.txt")
top = probs[0].topk(10)
for i, idx in enumerate(top.indices):
    print(f"{i+1}. {labels[idx]['common']}: {top.values[i]:.3f}")
```

### ONNX

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("exports/geomodel_fp16.onnx")

inputs = np.array([[52.5, 13.4, 22.0]], dtype=np.float32)  # (batch, 3)
probs = session.run(None, {"input": inputs})[0]             # (batch, n_species)

labels = load_labels()
top_k = 10
top_indices = np.argsort(probs[0])[::-1][:top_k]
for i, idx in enumerate(top_indices):
    print(f"{i+1}. {labels[idx]['common']}: {probs[0][idx]:.3f}")
```

### TFLite

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="exports/geomodel_fp16.tflite")
interpreter.allocate_tensors()

input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

inputs = np.array([[52.5, 13.4, 22.0]], dtype=np.float32)
interpreter.set_tensor(input_detail["index"], inputs)
interpreter.invoke()
probs = interpreter.get_tensor(output_detail["index"])       # (1, n_species)

labels = load_labels()
top_k = 10
top_indices = np.argsort(probs[0])[::-1][:top_k]
for i, idx in enumerate(top_indices):
    print(f"{i+1}. {labels[idx]['common']}: {probs[0][idx]:.3f}")
```

!!! tip "Batch inference"
    All formats support batched inputs. Stack multiple `[lat, lon, week]` rows into a single `(N, 3)` array to predict many locations at once.
