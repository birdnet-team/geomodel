# Export and inference

## Exporting

```bash
python convert.py --checkpoint release/BirdNET+_Geomodel_V4.0_Global_17K.pt --outdir exports \
    --formats onnx onnx_fp16 onnx_int8 torchscript
```

| format | file | contents |
|---|---|---|
| `onnx` | `geomodel.onnx` | float32 |
| `onnx_fp16` | `geomodel_fp16.onnx` | weight matrices stored as float16, computed in float32; scores within 0.01 of FP32 |
| `onnx_int8` | `geomodel_int8.onnx` | weight matrices stored as int8 (one scale per output channel), computed in float32; the smallest file, scores within about 0.2 of FP32 and usually much closer |
| `tflite` | `geomodel.tflite` | TensorFlow Lite, float32 |
| `tflite_fp16` | `geomodel_fp16.tflite` | TensorFlow Lite with float16 weights |
| `tflite_int8` | `geomodel_int8.tflite` | dynamic-range INT8 (activations rounded to 8 bits as well) |
| `tf` | `saved_model/` | TensorFlow SavedModel |
| `torchscript` | `geomodel.pt` | traced TorchScript; returns logits unless `--torchscript_sigmoid` |

The apps ship the INT8 ONNX file. Its weights take one byte each, but nothing else
is rounded: ONNX Runtime's own dynamic quantization, and TFLite's dynamic-range INT8,
also round the activations to 8 bits, which pushes some scores far up at places far
from any data (a house spider at 0.9 in the middle of the Pacific in our tests).
Storing int8 weights and computing in float32 avoids that at the same file size.

Each export is run on a fixed set of places and weeks and compared with PyTorch; it
fails if the largest score difference exceeds the format's tolerance (1e-4 for
float32, 0.01 for float16 weights, 0.25 for int8 weights). The label file next to the
checkpoint and the model license are copied to the output folder.

The release files on GitHub are made with the
[BirdNET model converter](https://github.com/birdnet-team/model-converter), which uses
the same model definition and adds LiteRT files with built-in operators only and
Core ML packages.

## What the exports compute

All formats take one float32 tensor of shape `(batch, 3)` with the columns latitude
(degrees), longitude (degrees) and week (1-48), and return `(batch, n_species)`
scores in the order of the label file. The score curve, the floors for humans and
domestic animals and the cap for domestic animals are part of the graph
([Scores](scores.md)), so an export returns exactly what `predict.py` prints. For a
whole-year list, run all 48 weeks and take the maximum per species.

The label file has one line per output: BirdNET id, scientific name and common
name, tab-separated.

## Running them

```python
import numpy as np
import onnxruntime as ort

labels = [line.rstrip('\n').split('\t') for line in open('exports/labels.txt', encoding='utf-8')]
session = ort.InferenceSession('exports/geomodel_int8.onnx')
scores = session.run(None, {'input': np.array([[50.83, 12.92, 22]], np.float32)})[0][0]
for i in np.argsort(-scores)[:10]:
    print(f'{scores[i]:.3f}  {labels[i][2]}')
```

```python
import torch

model = torch.jit.load('exports/geomodel.pt', map_location='cpu').eval()
with torch.no_grad():
    scores = torch.sigmoid(model(torch.tensor([[50.83, 12.92, 22.0]])))
```

In Python with the repository, `model.io.load` opens any checkpoint (release,
student, teacher, or a V3 release) and returns an object with `.species`,
`.scores(lat, lng, week)` and, for models that predict encounter rates,
`.rates(...)`:

```python
from model.io import load

model = load('release/BirdNET+_Geomodel_V4.0_Global_17K.pt')
scores = model.scores([50.83], [12.92], [22])[0]
```

## predict.py

```bash
python predict.py --checkpoint release/BirdNET+_Geomodel_V4.0_Global_17K.pt --lat 50.83 --lon 12.92 --week 22
python predict.py --checkpoint ... --lat 50.83 --lon 12.92 --week -1 --threshold 0.05 --top-k 300
python predict.py --checkpoint ... --common-species 5000 --taxonomy taxonomy.json --output common.txt
```

`--week -1` gives the whole year. `--common-species N` builds a list of N species
that covers the world evenly: mean scores over a 5° grid and four weeks for 20
regions, then, round by round, the next four birds and one other species from each
region's ranking.
