import geopandas as gpd
import gzip
import logging
import os
from collections import defaultdict

import h3
import pandas as pd
from tqdm import tqdm

# Columns read from the processed GBIF CSV
GBIF_REQUIRED_COLUMNS = ['latitude', 'longitude', 'taxonKey', 'week', 'class']

# Number of BirdNET weeks per year
NUM_WEEKS = 48


def estimate_gzip_rows(file_path, sample_rows=10000):
    """Estimate total rows in a gzipped CSV by sampling compressed byte ratios."""
    compressed_size = os.path.getsize(file_path)

    with gzip.open(file_path, 'rb') as f:
        f.readline()  # skip header
        start_pos = f.fileobj.tell()

        for _ in range(sample_rows):
            if not f.readline():
                break

        end_pos = f.fileobj.tell()

    compressed_sample_size = end_pos - start_pos
    if compressed_sample_size > 0:
        return max(int((compressed_size - start_pos) / (compressed_sample_size / sample_rows)), 1)
    return 0


def combine_geodata_and_gbif(geodata_path, gbif_processed_path, output_path, valid_classes):
    """
    Combine geographical data with processed GBIF data.

    Reads an H3-indexed GeoParquet and a processed GBIF CSV, maps each GBIF
    observation to its H3 cell and week, and writes a combined parquet with
    per-week species lists.
    """
    gdf = gpd.read_parquet(geodata_path)

    # Determine H3 resolution from data
    h3_res = h3.get_resolution(gdf.iloc[0]['h3_index'])
    logging.info(f"H3 resolution {h3_res}")

    valid_h3_cells = set(gdf['h3_index'])
    valid_classes_set = set(valid_classes)
    missing_cells = set()

    # Accumulate species per (cell, week) — sets for automatic deduplication
    cell_week_species = defaultdict(set)

    estimated_rows = estimate_gzip_rows(gbif_processed_path)

    with gzip.open(gbif_processed_path, 'rt', encoding='utf-8') as f:
        with tqdm(total=estimated_rows, desc="Processing GBIF data") as pbar:
            for chunk in pd.read_csv(f, chunksize=100000, usecols=GBIF_REQUIRED_COLUMNS):
                chunk_size = len(chunk)

                # Filter to valid classes
                chunk = chunk[chunk['class'].isin(valid_classes_set)]

                if not chunk.empty:
                    # Compute H3 cells
                    chunk = chunk.copy()
                    chunk['h3_cell'] = [
                        h3.latlng_to_cell(lat, lon, h3_res)
                        for lat, lon in zip(chunk['latitude'].values, chunk['longitude'].values)
                    ]

                    # Track missing cells (log once per cell)
                    mask = chunk['h3_cell'].isin(valid_h3_cells)
                    new_missing = set(chunk.loc[~mask, 'h3_cell'].unique()) - missing_cells
                    if new_missing:
                        logging.warning(f"{len(new_missing)} new H3 cell(s) not in geodata")
                        missing_cells.update(new_missing)

                    # Accumulate species
                    for h3_cell, week, taxon in zip(
                        chunk.loc[mask, 'h3_cell'].values,
                        chunk.loc[mask, 'week'].values,
                        chunk.loc[mask, 'taxonKey'].values,
                    ):
                        cell_week_species[(h3_cell, int(week))].add(taxon)

                pbar.update(chunk_size)

    # Build index mapping and assign week columns
    h3_to_idx = dict(zip(gdf['h3_index'], gdf.index))

    for week in range(1, NUM_WEEKS + 1):
        gdf[f'week_{week}'] = [[] for _ in range(len(gdf))]

    for (h3_cell, week), species_set in cell_week_species.items():
        gdf.at[h3_to_idx[h3_cell], f'week_{week}'] = list(species_set)

    if missing_cells:
        logging.info(f"Total {len(missing_cells)} H3 cell(s) from GBIF not found in geodata")

    gdf.to_parquet(output_path, index=False)
    logging.info(f"Saved combined dataset to {output_path}")


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Combine geographical data with processed GBIF data.')
    parser.add_argument('--geodata', type=str, default='./data/global_350km_ee.parquet',
                        help='Path to the geographical data parquet file')
    parser.add_argument('--gbif', type=str, default='./outputs/gbif_processed.gz',
                        help='Path to the gzipped processed GBIF file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output parquet file. If not provided, generated from geodata filename.')
    parser.add_argument('--valid_classes', type=str, nargs='+', default=['Aves', 'Mammalia', 'Amphibia'],
                        help='Valid taxonomic classes to include from GBIF data')
    args = parser.parse_args()

    output = args.output
    if output is None:
        geodata_filename = os.path.basename(args.geodata).replace('.parquet', '_gbif.parquet')
        output = os.path.join('./outputs', geodata_filename)
        logging.info(f"No output file provided. Using default: {output}")

    combine_geodata_and_gbif(args.geodata, args.gbif, output, args.valid_classes)