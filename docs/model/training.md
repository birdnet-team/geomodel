# Training

Training has two steps. A **teacher** learns encounter rates from the records,
using coordinates, week and habitat. A **student**, the released model, learns the
teacher's predictions from coordinates and week alone.

```bash
python train_teacher.py data/train.npz runs/teacher --tag teacher --habitat --seed 1 \
    --benchmark data/benchmark/bench_20k --habitat-table data/ee/habitat_res5.parquet

python distill.py data/train.npz runs/teacher/teacher.pt data/ee/habitat_res5.parquet runs/student \
    --tag student --population data/ee/population_res5.parquet --ubiquitous data/ubiquitous.csv \
    --scale 1.1 --film 64 --bottleneck 288 --harmonics 16 --envelope 4 --ocean --steps 150000 \
    --benchmark data/benchmark/bench_20k --int8-check

python package.py runs/student/student.pt taxonomy.csv data/ubiquitous.csv release \
    --name BirdNET+_Geomodel_V4.0_Global_17K --nonbird-f0 0.001
```

The training file is built with naturalist effort (`pipeline.gbif_targets --effort
naturalist`, [GBIF records](../pipeline/gbif.md#effort)); `--ocean` and the separate
non-bird anchor (`--nonbird-f0 0.001`, [Scores](scores.md#a-separate-anchor-for-non-birds))
are part of the release recipe.

On one RTX 4090 the teacher takes about 25 minutes and the student about 35; the
benchmark evaluation at the end of each adds about ten minutes on the CPU. The
training file is held in memory (about 2 GB for the full data plus 3-4 GB during
batching).

## Why two models

Apps only know where and when, so the released model cannot take habitat as an
input. But habitat is what lets a model predict a species in places nobody has
recorded it: a teacher that has seen a warbler in many mountain forests will
predict it in the next mountain forest. Distillation hands this knowledge to a
model that only needs coordinates. The student is trained on the teacher's
predictions for every land cell on Earth and every week, not only where records
exist, so it fills the gaps the way the teacher would.

## Teacher

### Data and loss

A training row is one cell, week and group (birds, or all other species) with its
observer-day effort and the encounter rates of the species recorded there
([Training file](../pipeline/dataset.md)). The loss is binary cross-entropy against
those rates, summed over the species of the row's group and weighted per row by
effort:

$$
L = \frac{\sum_{\text{rows}} w \cdot \sum_{s \in \text{group}} \text{BCE}(z_s, r_s)}{\sum_{\text{rows}} w},
\qquad w = \frac{n}{n + c}
$$

Binary cross-entropy against a rate is the binomial log-likelihood per observer-day,
so the sigmoid of the output learns the rate itself. The weight saturates: a row
backed by one observer-day counts a little, a row with fifty almost fully, and a cell
with fifty thousand checklists does not outweigh a thousand cells with fifty
(`--c 5`). Species of the other group are masked out.

### Optimization

AdamW (learning rate 1e-3, weight decay 1e-4), one-cycle schedule, batches of 1,024
rows, gradient clipping at 5, six epochs. The epoch with the lowest validation loss
is kept; training stops two epochs after the last improvement. The output bias
starts at -5 (rate 0.007), since most species are absent from most rows.

### Evaluation

With `--benchmark`, the teacher is scored on the eBird benchmark at the end
([Benchmark](../evaluation/benchmark.md)); habitat features at the benchmark cells
come from `--habitat-table`. In that mode the test blocks join the training data,
because the benchmark is the test set. `--holdout-km 150` removes every training
cell within 150 km of a benchmark cell for a strictly spatial test; give the
student the same `--holdout-km` (with `--benchmark`), so those records also stay out
of its range envelope and its list of marine species. Without
`--benchmark` the script reports, per group, log loss, top-10 capture, recall of
regular species and rank correlation on the test blocks, next to a baseline that
predicts each species' average rate everywhere.

## Student

### The pool

Each step draws 2,048 cells from the pool, every land cell of the habitat table
(about 540,000) plus every cell of the training file (sea cells with records
included), with a random week each. The teacher predicts every species there, and
the student is fitted to those predictions with binary cross-entropy. No records
are used in this step.

### What is added to the teacher's predictions

**Range envelope** (`--envelope 4`). A species' target is set to zero more than four
degrees (1° bins: about 440 km north-south, less east-west away from the equator,
about 220 km at 60° N) from any of its records. Habitat matching alone
would put a species on every continent with a similar climate: a European forest
bird in the forests of Japan. The envelope keeps the teacher's habitat knowledge
local.

**Humans and domestic animals** (`--ubiquitous`, `--population`). Their targets get a
floor that grows with population density and built-up area, from 0.30 (humans) or
0.25 (domestic animals) on empty land to 0.50 in towns ([Scores](scores.md)).

**Ocean** (`--ocean`). Open-ocean cells (H3 resolution 3, more than 25 km from any
land) join the pool, and at sea every species except the marine ones is set to
zero. Marine species are those recorded in at least five open-ocean training rows.
Without this, the student interpolates freely over water and land species near a
coast spill hundreds of kilometers offshore.

### Optimization

AdamW (learning rate 1e-3, weight decay 1e-5), one-cycle schedule, 150,000 steps.
The output bias starts at -6. The checkpoint holds the weights, the species list
(BirdNET ids), their eBird codes and the arguments.

### INT8

The apps ship the INT8 ONNX file: weight matrices stored as int8 with one scale per
output channel, arithmetic in float32 ([Export](export.md)). This can be emulated
exactly in PyTorch by rounding the weights, and `--int8-check` scores the student
that way next to the float32 result. The difference in GeoScore is a few tenths.

## Ablation switches

Options that the release recipe does not use stay available for comparisons:

| script | option | what it does |
|---|---|---|
| `train_teacher.py` | `--link cloglog` | rate = 1 - exp(-exp(z)), the link of a Poisson abundance model |
| | `--pos-weight` | scales the loss of entries with a positive rate |
| | `--gamma-pos`, `--gamma-neg` | focal down-weighting of easy entries (asymmetric loss) |
| | `--rank-weight` | adds a list-wise term that rewards putting common species on top |
| | `--c` | effort at which a row has half weight |
| | `--no-coords` | habitat and week only |
| | without `--habitat` | coordinates and week only (no distillation needed) |
| | `--week-harmonics`, `--coord-harmonics`, `--scale` | network shape |
| `distill.py` | `--envelope 0` | no range envelope |
| | without `--ocean` | no ocean rule |
| | `--env-aux` | predict the habitat as an auxiliary task |
| | `--habitat-head` | habitat-species head with a learned gate |
| | `--qat-frac` | INT8-aware training (weights and inputs) at the end |
| `pipeline.gbif_targets` | `--effort group` | non-bird effort from the species' own group |
| | `--effort modeled` | non-bird effort from observer-days with a modeled species |
| | `--half-life` | recency weighting (large value = none) |

## Reproducibility

`--seed` fixes the batch order and initialization. Two teachers trained with
different seeds differed by 0.9 GeoScore points on the benchmark (77.5 and 76.6;
early stopping picked epoch 3 in one run and epoch 2 in the other), while two
students distilled from the same teacher with different seeds differed by 0.02.
Differences between teachers smaller than about a point are noise; compare
recipes on more than one teacher seed.
