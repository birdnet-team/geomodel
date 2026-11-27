# Taxon groups

```bash
python -m evaluation.groups release/BirdNET+_Geomodel_V4.0_Global_17K.pt data/benchmark/bench_20k --taxonomy taxonomy.json
```

Encounter rates of different taxon groups are comparable only if they are measured
against comparable effort ([GBIF records](../pipeline/gbif.md#effort)). When they are
not, small groups float to the top of every list. This check shows how the groups
share the top of the species lists.

At 400 benchmark places and four weeks (4, 16, 28, 40), per group:

| column | meaning |
|---|---|
| `outputs` | species of the group in the model |
| `per_place_ge_0_5`, `per_place_ge_0_9` | species scoring at least 0.5 / 0.9 per place |
| `top5_mean` | mean score of the group's five highest species per place |
| `share_of_top20` | share of each place's top 20 that the group takes |

It then lists the highest-scoring species for a few named places (a European city,
New York, Nairobi, Sydney, Manaus and the open North Sea), which makes problems
easy to see: a crab or a carp on top of the list in an inland city means the
non-bird rates are out of scale.
