# Status and Trends

```bash
python -m evaluation.status_trends cache ST_DIR data/st_2023_0p5.npz
python -m evaluation.status_trends score CHECKPOINT data/st_2023_0p5.npz --taxonomy taxonomy.csv
```

[eBird Status & Trends](https://science.ebird.org/en/status-and-trends) models the
relative abundance of a species from eBird checklists with its own covariates and
effort correction, and gives one map per week. It is an independent reference for a
species' range and its timing, also in places with few observers. Maps exist for a
few thousand species; the cache holds whichever species you downloaded (the
reference values below use 11 migrants and residents).

`cache` reads the median-abundance downloads (`{code}_abundance_median_2023.zip`, 52
weekly rasters at 3 km) and averages them onto a regular grid (0.5° by default).
Each model week takes the Status & Trends week closest to its middle. It needs
`rasterio` and `pyproj`. The downloads themselves are subject to eBird's terms and
are not part of the repository.

`score` evaluates the model at every grid cell with Status & Trends data for all 48
weeks and reports per species:

| measure | meaning |
|---|---|
| `range_auc` | how well the model separates Status & Trends presence (abundance > 0) from absence, over space and season |
| `spatial_rho` | rank correlation of model and abundance over the occupied cells of a week, averaged over weeks |
| `season_rho` | rank correlation over the 48 weeks per occupied cell, median over cells: are arrival and departure timed right? |

All three are rank measures within a species, so the plain network output is used.

Reference (V3.0.4, mean over the species where a measure is defined; season rho is
undefined for the Arctic Tern): range AUC 0.957, spatial rho 0.26, season rho 0.64.
