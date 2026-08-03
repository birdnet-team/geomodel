# Combining Data

`utils/combine.py` joins the H3 environmental grid (from Stage 1) with the processed GBIF observations (from Stage 2) to produce a training-ready dataset.

## How It Works

1. **Load the H3 grid** — reads the GeoParquet with environmental features
2. **Stream GBIF observations** — reads the processed CSV in chunks
3. **Map observations to cells** — each observation's (lat, lon) is mapped to its containing H3 cell using `h3.latlng_to_cell()`
4. **Aggregate by week** — for each cell, observations are grouped by BirdNET week number (1–48), producing a list of species codes per week
5. **Write outputs** — combined parquet and a taxonomy CSV

## CLI Options

```bash
python utils/combine.py \
    --geodata data/global_50km_ee.parquet \
    --gbif ./outputs/gbif_processed.csv.gz \
    --output ./outputs/combined.parquet \
    --valid_classes Aves Mammalia Amphibia \
    --workers 16
```

| Flag | Description |
|---|---|
| `--geodata` | H3 GeoParquet from `geoutils.py` |
| `--gbif` | Processed GBIF CSV from `gbifutils.py` |
| `--output` | Output path for combined parquet |
| `--valid_classes` | Taxonomic classes to include (default: all) |
| `--workers` | Parallel worker processes for H3 cell computation (default: 1) |

## Output Files

### Combined Parquet

Each row is an H3 cell with:

| Columns | Description |
|---|---|
| `h3_index` | H3 cell identifier |
| `geometry` | Cell polygon |
| Environmental columns | `elevation_m`, `temperature_c`, etc. |
| `week_1` … `week_48` | List of species codes observed in that week |

### Taxonomy CSV

Auto-generated alongside the parquet (with `_taxonomy.csv` suffix):

| Column | Description |
|---|---|
| `species_code` | eBird species code or iNat ID |
| `scientificName` | Binomial scientific name |
| `commonName` | Common name (if available) |

## Species Remapping (recent taxonomic splits)

Observation databases (GBIF, iNaturalist) often keep recording occurrences
under a **pre-split lumped scientific name** for years after a taxonomic split.
Because observations are attributed by `verbatimScientificName`, this silently
routes the wrong species code — for example, the eBird 2024 split of
*Setophaga petechia* into *S. aestiva* (American Yellow Warbler) and
*S. petechia* (Mangrove Yellow Warbler) leaves the vast majority of North
American records still labeled *Setophaga petechia*, which would otherwise be
attributed to the tropical Mangrove form.

`species-data/species_remap.csv` fixes this at the source. Each row redirects a
verbatim scientific name to a canonical species code before observations are
aggregated:

| Column | Description |
|---|---|
| `from_sci_name` | Verbatim scientific name as it appears in GBIF/iNat data |
| `from_species_code` | (Optional) resolved code the split records currently carry |
| `to_species_code` | Canonical species code to attribute those records to |
| `note` | Rationale for the remap (free text) |

The remap is loaded automatically by `utils/taxonomy.py: TaxonomyManager`
(default path `species-data/species_remap.csv`). Both species remain valid
vocabulary entries — only the attribution of the ambiguous verbatim name
changes. Add a row whenever a split leaves observations stranded under an
outdated name.

**Correcting an already-combined parquet without re-running combine.py.**
The optional `from_species_code` column also defines a direct code→code
redirect that `train.py` (and the autotuner) apply when loading the parquet, so
a split can be corrected at train time. For the Yellow Warbler example,
`from_species_code=yelwar, to_species_code=yelwar1` rewrites every `yelwar`
label to `yelwar1` during flattening. Controlled by `--species_remap`
(default `species-data/species_remap.csv`; pass `''` to disable). The data cache
is keyed on the remap file contents, so edits trigger a reprocess automatically.
