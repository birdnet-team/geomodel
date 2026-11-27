# Scores

## What a score means

A score ranks species by how often they are encountered at a place and time: very
common species on top, rare ones at the bottom, and the course of the score over
the year follows migration. It is not a probability of presence and not a range map.
Species that are easy to notice rank above secretive ones of the same abundance,
because they are encountered more often; for a filter of audio detections that is
the right order.

The networks predict an **encounter rate** $r$, the share of checklists or
observer-days that report the species. Rates span six orders of magnitude, from
0.5 for a blackbird in a European garden to 0.0001 for a once-a-decade vagrant. Apps
need a number between 0 and 1 with a sensible line at 0.1, so releases map the rate
through a **score curve**:

$$
\text{logit}(\text{score}) = \text{logit}(a) + s \cdot (\log_{10} r - \log_{10} f_0)
$$

with $f_0 = 0.003$, $a = 0.1$ and $s = 3$: a species reported on 0.3 % of checklists
scores 0.1, and every factor of ten in rate moves the score by three logit units.

| encounter rate | score |
|---|---|
| 0.0001 (1 in 10,000) | 0.001 |
| 0.001 | 0.03 |
| 0.003 | 0.10 |
| 0.01 | 0.35 |
| 0.03 | 0.69 |
| 0.1 | 0.91 |
| 0.3 | 0.98 |

Species seen on one checklist in a hundred sit in the middle of the range, species
seen on every third checklist at the top, and the long tail of rare visitors well
below the line. The curve is
stored in the release checkpoint (`score_curve`) and applied in the exported graph,
so every format returns the same scores. `model.io.rate_to_score` and
`score_to_rate` convert in Python.

### A separate anchor for non-birds

With `pipeline.gbif_targets --effort naturalist` or `modeled`, non-bird rates are
shares of all naturalist activity in a place, and species that are heard far more
often than photographed (many frogs and insects) end up with low rates.
`package.py --nonbird-f0 0.001` gives non-birds their own anchor: a non-bird reported
on 0.1 % of naturalist observer-days then scores 0.1. Humans and domestic animals
keep the common anchor, because their floors were trained on it. The curve stores
one anchor per output (`f0_species`); `convert.py`, `model.io` and the
[BirdNET model converter](https://github.com/birdnet-team/model-converter) apply it.
Without the option all species share the same anchor. The V4 release uses it: with
the common anchor, much-heard frogs such as the spring peeper stay below 0.1 (0.07 in
New York in April); with the non-bird anchor they reach 0.25, while non-birds take
about 6 % of each place's top 20 (40 % before naturalist effort, 1 % without the
anchor). Much-photographed insects rise as well, which does no harm in a filter for
sounds.

## Humans and domestic animals

The apps use the human class to recognize speech: if someone holds a phone, there is
a human. Domestic animals can turn up wherever people live. Neither follows from
observation records. They are listed in `data/ubiquitous.csv` and handled with
floors and caps:

| | during training (student targets) | in the release |
|---|---|---|
| human | at least 0.30, up to 0.50 where many people live | floor 0.30 |
| domestic animals | at least 0.25, up to 0.50 where many people live | floor 0.25, cap 0.50 |

"Many people" means about 100 inhabitants per km² or 20 % built-up area
([population](../pipeline/earthengine.md#population)). The exported model only sees
coordinates and week, so the release keeps the constant part of the floor; the
population-dependent part is learned by the student. Domestic animals are capped at
0.5 so that they pass everywhere without topping a list. Species in the file without
any records (the domestic dog) are added as outputs.

## Choosing a threshold

The apps hide detections of species scoring below 0.1 by default. On the 2025 app
study (see [App filter study](../evaluation/filter_study.md)) this rejects about one
in a hundred confident detections and a quarter of the unlikely second and third
guesses of the audio model. A lower line (0.05) rejects fewer true detections and
fewer false ones; a higher line the opposite.
