# Upgrading from V3

This page is for apps and libraries that use the V3 geomodel (V3.0.x releases). The
input is the same in V4. The meaning of the outputs and their labels change.

## What stays

- One float32 input tensor `[batch, 3]`: latitude, longitude, week (1-48, four per
  month). For the whole year, run all 48 weeks and take the maximum per species, as
  `predict.py --week -1` does. Do not pass week 0 or -1 to the model files. The week
  is read as a position on a circle, so 0 is the same as week 48 (late December) and
  -1 the same as week 47. V3 files had learned a whole-year answer for week 0, V4
  files have not.
- One value per species, in label-file order, between 0 and 1.
- TorchScript returns logits; ONNX and TFLite return the final values.

## What changes

| | V3 | V4 |
|---|---|---|
| species | 14,082 | 16,960 |
| label ids | eBird codes for birds, iNaturalist ids for others | BirdNET ids (`BN…`) for all |
| label file | `id <TAB> scientific name <TAB> common name` | same layout, BirdNET ids |
| output | "was the species recorded here", from union labels | score from the encounter rate |
| humans, domestic animals | as learned | humans ≥ 0.30, domestic animals 0.25-0.50 everywhere |
| INT8 file | TFLite dynamic-range INT8 | ONNX with int8 weights, float32 arithmetic |

### Ids

Every V4 output is a species of the BirdNET taxonomy and is labeled with its BirdNET
id, the same id the BirdNET audio models use. Join geomodel and audio results by id,
not by name or eBird code. The taxonomy
([Species](../pipeline/taxonomy.md)) maps BirdNET ids to eBird codes, iNaturalist
ids and names.

### Scores

A V3 output was close to 1 wherever a species had been recorded at all,
including a single old vagrant record. A V4 score follows how often the species is
encountered ([Scores](scores.md)): common species near 1, regular ones in the
middle, rare visitors near 0. Lists filtered at the same line are therefore shorter
where many people record. On the eBird benchmark, at 0.1, the V4.0 candidate passes
205-227 bird species per cell and week in well-observed cells against 295-304 for
V3.0.4, and 210 against 239 in thinly observed ones, while keeping 96-97 % of the
regular species. Rare visitors in well-observed cells pass about half as often.
(Numbers for the V4.0 candidate; they will be updated for the final release.)

### Threshold

V3 apps used lines around 0.1-0.15 on V3 outputs. For V4, the recommended line is
0.1, which corresponds to a species reported on about 0.3 % of checklists. On the 2025 app
study, V4 at 0.1 rejects slightly fewer confident detections than V3 at 0.1 and
removes as many unlikely ones ([App filter study](../evaluation/filter_study.md)).

### Humans and domestic animals

In V4, humans and domestic animals no longer depend on records. Humans always pass
(their score never drops below 0.30), so apps that use the human class for speech
detection can rely on it anywhere, including at sea ([Scores](scores.md#humans-and-domestic-animals)).

### INT8

The V4 INT8 file is ONNX (`…_INT8.onnx`). Weights are stored in 8 bits, all
arithmetic is float32, and scores stay within about 0.2 of the FP32 model. The V3
TFLite INT8 file also rounded activations, which moved some scores by a lot at
places far from the data ([Export](export.md)). FP32 files are provided in ONNX, TFLite and TorchScript, FP16 files in ONNX
and TFLite.

## Checking an integration

Run the same place and week through your integration and through `predict.py`. The
scores should match to about 1e-4 for FP32 files, 0.01 for ONNX FP16 (somewhat more
for TFLite FP16) and 0.2 for INT8:

```bash
python predict.py BirdNET+_Geomodel_V4.0_Global_17K.pt --lat 50.83 --lon 12.92 --week 22 --top-k 20
```

## Old checkpoints

V3.0.x release checkpoints load in `predict.py`, `convert.py` and the evaluation
scripts. Their outputs are the network's own values, exactly as in their
ONNX and TFLite exports (V3.0.4: within 3e-5 of its FP32 ONNX file), with V3 ids and
the V3 label file (`..._Global_14K_Labels.txt` is found next to `..._Global_14K_FP32.pt`).
V3's `predict.py` also set non-marine species to 0 at open-sea points. That step was
never part of the V3 exports, and this repository's `predict.py` does not apply it.
