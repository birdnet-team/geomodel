# App filter study

```bash
python -m evaluation.filter_study CHECKPOINT geofilter_pairs.parquet --taxonomy taxonomy.csv
python -m evaluation.filter_study BirdNET_Geomodel_V3.0.4.pt geofilter_pairs.parquet --labels BirdNET_Geomodel_V3.0.4_labels.txt
```

The geomodel exists to filter audio detections, so the most direct test is what it
would remove from real recordings. The study uses about 100,000 BirdNET app
recordings from 2025, sampled evenly across the world and re-classified with the
BirdNET+ V3 audio model, with the top three bird predictions of each.

There are no verified identifications, so two proxies stand in:

- **likely true**: rank-1 predictions with classifier confidence of at least 0.9
  that agree with the species the app showed (about 48,000);
- **likely false**: rank-2 and rank-3 predictions with confidence below 0.3
  (about 174,000).

Per score line the script reports the share of likely-true detections the geomodel
would remove (false rejections, the costly error), the share of likely-false ones
it removes, and the share removed at each rank. Predictions of species the model
does not have pass unfiltered and are left out.

The recordings are app data and are not public; the script documents the expected
columns (`rank`, `sci`, `score`, `latitude`, `longitude`, `week`, `app_scientific`).

Reference (V3.0.4, line 0.10): 1.2 % false rejections, 27.1 % of likely-false
predictions removed; removed at rank 1 / 2 / 3: 3.5 / 23.5 / 26.4 %.
