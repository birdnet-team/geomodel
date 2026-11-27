# Seasonality

```bash
python -m evaluation.seasonality CHECKPOINT data/benchmark/bench_20k --taxonomy taxonomy.csv \
    --compare BirdNET_Geomodel_V3.0.4.pt --compare-labels BirdNET_Geomodel_V3.0.4_labels.txt
python -m evaluation.seasonality CHECKPOINT data/benchmark/bench_20k --taxonomy taxonomy.csv --species barswa --weeks 21 24
```

A filter that passes a migrant in the months it is away lets through exactly the
misidentifications it should catch. This check asks how often that happens, on the
eBird benchmark.

For every benchmark cell and every species whose frequency there reaches at least
5 % in some week:

- an **off-season week** is a week with at least 50 checklists and frequency 0, an
  observed absence;
- an **in-season week** is a week with frequency of at least 5 %.

Reported north and south of the equator: **leak**, the share of off-season weeks the
model scores at 0.1 or more (lower is better), and **recall**, the share of in-season
weeks it scores at 0.1 or more. Then the species with the most leaking weeks, or with
`--compare`, the species that leak most compared with the second model. With two
models only the species both have are counted.

Part of the leak is unavoidable: the weeks right before arrival and after departure
count as off-season if no checklist reported the species, while a smooth model
still scores it there. Comparisons between models are what matters.

`--species CODE --weeks FIRST LAST` looks at one species over a few weeks: cells with
at least ten checklists, grouped by the species' frequency there (absent, up to 1 %,
1-5 %, over 5 %), with the median score and the share of cells at 0.1 or more per
hemisphere. Barn Swallow in June (weeks 21-24) should stay below the line in the
south, where it is absent.

Reference (V3.0.4): leak 59.9 % north, 71.0 % south; recall 98.2 % and 96.9 %.
