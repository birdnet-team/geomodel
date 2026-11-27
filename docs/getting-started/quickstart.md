# Quick start

## Species list for a place

Download a release (checkpoint and label file) from the
[releases page](https://github.com/birdnet-team/geomodel/releases), then:

```bash
python predict.py BirdNET+_Geomodel_V4.0_Global_17K.pt --lat 50.83 --lon 12.92 --week 22
```

It prints the species scoring at least 0.1 there, highest first, with score,
BirdNET id, common and scientific name.

!!! note
    V4.0 is not released yet. Until it is, the latest release is V3.0.4. `predict.py`
    loads its checkpoint as well (`BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.pt` next to its
    label file), with V3 outputs and ids ([Upgrading from V3](../model/upgrading.md)).

Weeks run from 1 to 48, four per month. `--week -1` gives the whole year.
`--threshold` and `--top-k` change how much is shown.

## In your own code

The ONNX files need nothing but `onnxruntime`:

```python
import numpy as np
import onnxruntime as ort

labels = [line.rstrip('\n').split('\t') for line in open('BirdNET+_Geomodel_V4.0_Global_17K_labels.txt', encoding='utf-8')]
session = ort.InferenceSession('BirdNET+_Geomodel_V4.0_Global_17K_INT8.onnx')
scores = session.run(None, {'input': np.array([[50.83, 12.92, 22]], np.float32)})[0][0]
```

`scores[i]` belongs to `labels[i]` (BirdNET id, scientific name, common name). A
score of 0.1 corresponds to a species reported on about 0.3 % of checklists in the
area ([Scores](../model/scores.md)).

## Training your own model

The full pipeline, from a GBIF download to an exported model:

1. [Data pipeline](../pipeline/overview.md): records, species, encounter rates,
   habitat, training file, benchmark.
2. [Training](../model/training.md): teacher, student, release checkpoint.
3. [Export](../model/export.md): ONNX, TFLite, TorchScript.
4. [Evaluation](../evaluation/benchmark.md): eBird benchmark, seasonality, Status &
   Trends, app filter study, range maps.
