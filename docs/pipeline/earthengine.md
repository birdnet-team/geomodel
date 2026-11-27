# Habitat and population

The teacher model uses the habitat of each place to carry what it learns from
well-observed places to similar ones with few records. The habitat comes from
[Google Earth Engine](https://earthengine.google.com) (an account and
`earthengine authenticate` once per machine).

```bash
python -m pipeline.earthengine cells data/ee/cells_res5.txt
python -m pipeline.earthengine habitat data/ee/cells_res5.txt data/ee/habitat_parts
python -m pipeline.earthengine collect data/ee/habitat_parts data/ee/habitat_res5.parquet
python -m pipeline.earthengine population data/ee/cells_res5.txt data/ee/population_parts
python -m pipeline.earthengine collect data/ee/population_parts data/ee/population_res5.parquet
```

## Cells

`cells` lists all resolution-5 cells (~250 km²) on land plus about 45 km of sea
around the coasts: a resolution-4 cell counts as land if any of its 49 resolution-6
sub-cells has land at its center (`global_land_mask`, 1 km), which finds small
islands too; one ring of neighbors is added, and everything is split into
resolution-5 cells. About 690,000 cells.

## Habitat features

Every value is an **area mean over the hexagon** at about 1 km, not a sample at its
center: a cell on a coast is part sea, part land, and its features should say so.

| columns | layer |
|---|---|
| `elev`, `elev_sd` | GMTED2010 elevation, mean and standard deviation |
| `bio01` … `bio19` | WorldClim V1 bioclimatic variables |
| `tavg_01` … `tavg_12` | WorldClim V1 monthly mean temperature |
| `prec_01` … `prec_12` | WorldClim V1 monthly precipitation |
| `water_occ` | JRC Global Surface Water occurrence (0-100) |
| `canopy_h` | canopy height, NASA/JPL 2005 |
| `lc_01` … `lc_17` | MODIS MCD12Q1 (2022) land-cover class fractions, IGBP classes |
| `n_px` | number of 1 km climate pixels in the cell (0 = open water) |

Cells without climate data (open water) have empty climate columns; the training
file marks them with a 0 in its first feature column.

## Population

`pop_density` (people per km², GPWv4.11, 2020) and `built` (built-up fraction, GHSL,
2020). They set the minimum score of humans and domestic animals during training:
higher where more people live.

## Running it

Both fetch commands send the cells in chunks (`--chunk`) over a few parallel
requests (`--threads`) and write one `part_*.jsonl.gz` per chunk. Existing parts are
skipped, so an interrupted run resumes; a chunk that fails four times stops the run
with an error instead of leaving a silent gap. `collect` joins the parts into one
parquet table with the columns in alphabetical order (the training file depends on
a fixed order).
