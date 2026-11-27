# Range maps

```bash
python -m evaluation.range_maps CHECKPOINT maps/ --species "Hirundo rustica" "Ciconia ciconia" --taxonomy taxonomy.json \
    --compare BirdNET_Geomodel_V3.0.4.pt --compare-labels BirdNET_Geomodel_V3.0.4_labels.txt --compare-name V3.0.4
```

One animated GIF per species, one frame per week, showing the app score on a regular
grid (1° by default, 80° N to 60° S). Land is gray and only land is colored, with
color steps at 0.02, 0.05, 0.1, 0.25 and 0.5, so the default app line of 0.1 is one
of the edges. Each panel title gives the share of land at or above 0.1 that week.
With `--compare` the second model is drawn on the left on the same scale, which
makes changes between versions easy to see.

The land mask comes from the habitat table of [stage 4](../pipeline/earthengine.md)
(`--land`). Species are given by scientific name.

For the overview animation in the README (several species in one GIF), use
`scripts/demo_gif.py`.
