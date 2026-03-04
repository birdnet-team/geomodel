# GBIF Occurrence Processing

`utils/gbifutils.py` reads a raw GBIF Darwin Core Archive (DwC-A) zip file, applies quality filters, and writes a clean CSV of species observations.

## Obtaining GBIF Data

1. Go to [GBIF.org](https://www.gbif.org/) and navigate to **Occurrences**
2. Apply filters for your region and taxa of interest (e.g. class Aves, country, date range)
3. Download the results as a **Darwin Core Archive** (`.zip`)
4. The zip contains a tab-separated CSV — note the filename inside for the `--file` argument

## Filters Applied

The processing pipeline applies these filters in order:

1. **Drop incomplete records** — rows missing coordinates, date, taxonKey, or scientific name
2. **Taxonomy filter** — when `--taxonomy` is provided, keep only species present in the taxonomy file and add common names (most selective, applied first for speed)
3. **Taxonomic class filter** — keep only specified classes (default: Aves, Amphibia, Insecta, Mammalia, Reptilia)
4. **Binomial names only** — keep species with exactly two words in the name (skip subspecies, genera, etc.)

## Week Numbering

Dates are converted to **BirdNET week numbers** (1–48), where each week spans approximately 7.6 days. This matches BirdNET's internal temporal resolution and ensures week 48 covers the end of December.

## CLI Options

```bash
python utils/gbifutils.py \
    --gbif /path/to/gbif_archive.zip \
    --file occurrence.csv \
    --output ./outputs/gbif_processed.csv.gz \
    --taxonomy taxonomy.csv \
    --max_rows 10000000 \
    --workers 8
```

| Flag | Description |
|---|---|
| `--gbif` | Path to GBIF Darwin Core Archive zip |
| `--file` | Name of the CSV file inside the zip |
| `--output` | Output path for processed CSV (`.csv.gz` for compression) |
| `--taxonomy` | Path to taxonomy CSV for filtering and name enrichment |
| `--max_rows` | Maximum rows to process (for testing with large archives) |
| `--workers` | Number of parallel worker processes (default: `min(cpu_count-1, 8)`) |

## Parallel Processing

For large archives (100M+ rows), the script uses multiprocessing to speed up
parsing and filtering.  The main process reads 64 MB byte blocks from the zip
sequentially, then distributes each block to a pool of worker processes for
independent parsing, filtering, and CSV output.  Results are collected in order
and written to a single output stream.

On an 8-core machine, this typically achieves 4–6× speedup over single-threaded
processing.

## Output Format

A gzipped CSV with columns:

| Column | Description |
|---|---|
| `latitude` | Decimal latitude |
| `longitude` | Decimal longitude |
| `taxonKey` | GBIF taxonomic identifier |
| `verbatimScientificName` | Scientific name from the record |
| `commonName` | Common name (from taxonomy, if available) |
| `week` | BirdNET week number (1–48) |
| `class` | Taxonomic class (e.g. Aves) |
