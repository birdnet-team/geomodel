"""Combine H3 geodata with processed GBIF observations.

Maps each GBIF record to its H3 cell and week, producing a combined
parquet with per-week species lists and an accompanying taxonomy CSV.

Supports multiprocessing (``--workers``) to parallelize the expensive
per-row H3 cell computation across CPU cores.
"""

import geopandas as gpd
import gzip
import logging
import multiprocessing as mp
import os
from collections import defaultdict

import h3
import pandas as pd
from tqdm import tqdm

try:
    from utils.taxonomy import TaxonomyManager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.taxonomy import TaxonomyManager

# Columns read from the processed GBIF CSV
GBIF_REQUIRED_COLUMNS = ['latitude', 'longitude', 'taxonKey', 'verbatimScientificName', 'commonName', 'week', 'class']

# Number of BirdNET weeks per year
NUM_WEEKS = 48

# ---------------------------------------------------------------------------
# Worker state (set once per process via initializer)
# ---------------------------------------------------------------------------

_valid_h3_cells = None
_valid_classes = None
_h3_res = None
_taxonomy_manager = None


def _init_worker(valid_cells, classes, resolution, taxonomy_path=None):
    """Initializer for pool workers — sets shared read-only state."""
    global _valid_h3_cells, _valid_classes, _h3_res, _taxonomy_manager
    _valid_h3_cells = valid_cells
    _valid_classes = classes
    _h3_res = resolution
    if taxonomy_path:
        _taxonomy_manager = TaxonomyManager(taxonomy_path)


def _process_chunk(chunk):
    """Process a single DataFrame chunk in a worker process.

    Returns:
        Tuple of (cell_week_species dict, taxon_names dict, missing_cells set,
        chunk_size int).
    """
    chunk_size = len(chunk)

    # Filter to valid classes (convert chunk classes to lowercase to match taxonomy defaults)
    chunk = chunk[chunk['class'].str.lower().isin([c.lower() for c in _valid_classes])]
    if chunk.empty:
        return {}, {}, set(), chunk_size

    # Compute H3 cells
    lats = chunk['latitude'].astype(float).values
    lons = chunk['longitude'].astype(float).values
    h3_cells = [h3.latlng_to_cell(lat, lon, _h3_res) for lat, lon in zip(lats, lons)]
    # Ensure H3 cells are always strings (hex)
    h3_cells_hex = [c if isinstance(c, str) else h3.int_to_str(c) for c in h3_cells]
    chunk = chunk.assign(h3_cell=h3_cells_hex)

    # Filter to valid cells
    mask = chunk['h3_cell'].isin(_valid_h3_cells)
    new_missing = set(chunk.loc[~mask, 'h3_cell'].unique())
    matched = chunk[mask]

    if matched.empty:
        return {}, {}, new_missing, chunk_size

    # Use taxonomy if available
    if _taxonomy_manager:
        # Map taxonKey/sci_name to primary ID (eBird code or iNat ID)
        def _get_primary_id(row):
            sci = str(row['verbatimScientificName']).strip()
            tk = int(row['taxonKey']) if pd.notna(row['taxonKey']) else None
            return _taxonomy_manager.get_primary_id(sci, fallback_gbif_key=tk)
            
        matched = matched.assign(primary_id=matched.apply(_get_primary_id, axis=1))
        group_key = 'primary_id'
        
        # Collect metadata from taxonomy (using consistent schema: sci_name, com_name, species_code, class_name)
        taxon_names = {}
        for tk, sci, com, tax_class in zip(
            matched['taxonKey'].values,
            matched['verbatimScientificName'].values,
            matched['commonName'].values,
            matched['class'].values,
        ):
            meta = _taxonomy_manager.get_metadata_by_name(str(sci).strip())
            pid = _taxonomy_manager.get_primary_id(str(sci).strip(), fallback_gbif_key=int(tk) if pd.notna(tk) else None)
            
            if meta:
                taxon_names[pid] = {
                    'taxonKey': int(tk) if pd.notna(tk) else None,
                    'sci_name': meta['sci_name'],
                    'com_name': meta['com_name'],
                    'species_code': meta['species_code'],
                    'class_name': meta['class_name'],
                }
            else:
                taxon_names[pid] = {
                    'taxonKey': int(tk) if pd.notna(tk) else None,
                    'sci_name': str(sci),
                    'com_name': str(com),
                    'species_code': pid,
                    'class_name': str(tax_class).lower(),
                }
    else:
        group_key = 'taxonKey'
        taxon_names = {}
        taxa = matched.drop_duplicates('taxonKey')
        for tk, sci, com in zip(
            taxa['taxonKey'].values,
            taxa['verbatimScientificName'].values,
            taxa['commonName'].values,
        ):
            pid = str(int(tk)) if pd.notna(tk) else str(sci)
            taxon_names[pid] = {
                'taxonKey': int(tk) if pd.notna(tk) else None,
                'sci_name': str(sci),
                'com_name': str(com),
                'species_code': pid,
                'class_name': 'unknown',
            }

    # Accumulate species per (cell, week) via groupby
    cell_week_species = {}
    for (cell, week), species in matched.groupby(
        ['h3_cell', 'week'],
    )[group_key].agg(set).items():
        cell_week_species[(cell, int(week))] = species

    return cell_week_species, taxon_names, new_missing, chunk_size


def estimate_gzip_rows(file_path, sample_rows=10000):
    """Estimate total rows in a gzipped CSV by sampling compressed byte ratios."""
    compressed_size = os.path.getsize(file_path)

    with gzip.open(file_path, 'rb') as f:
        f.readline()  # skip header
        start_pos = f.fileobj.tell()
        line_count = 0
        while line_count < sample_rows:
            line = f.readline()
            if not line:
                break
            line_count += 1
        end_pos = f.fileobj.tell()

    if line_count == 0:
        return 0
    bytes_per_row = (end_pos - start_pos) / line_count
    return int(compressed_size / bytes_per_row) if bytes_per_row > 0 else 0


def combine_data(h3_path, gbif_path, output_path, resolution=5, workers=1, classes=None, taxonomy_path=None):
    """Combine H3 environmental data with GBIF occurrences."""
    if classes is None:
        # If no classes provided, dynamically determine them from taxonomy
        if taxonomy_path:
            try:
                temp_tax = pd.read_csv(taxonomy_path)
                classes = temp_tax['class_name'].unique().tolist()
                logging.info(f"Using classes from taxonomy: {classes}")
            except Exception:
                classes = ['Aves']
        else:
            classes = ['Aves']
    
    logging.info(f"Loading H3 environmental data from {h3_path}")
    h3_df = pd.read_parquet(h3_path)
    valid_h3_cells = set(h3_df['h3_index'].values)
    logging.info(f"Loaded {len(valid_h3_cells)} valid H3 cells")

    # Estimate total rows for progress bar
    total_est = estimate_gzip_rows(gbif_path) if gbif_path.endswith('.gz') else None

    # Initialize shared state for multiprocessing
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(valid_h3_cells, classes, resolution, taxonomy_path)
    )

    # Global counters
    total_processed = 0
    missing_cells = set()
    taxon_metadata = {}
    # Nested mapping: (cell_h3, week) -> set of species IDs
    global_species = defaultdict(set)

    logging.info(f"Processing GBIF data from {gbif_path} with {workers} workers")
    
    # Read in chunks to keep memory usage balanced
    chunksize = 100000
    reader = pd.read_csv(gbif_path, chunksize=chunksize, usecols=GBIF_REQUIRED_COLUMNS)

    with tqdm(total=total_est, desc="Rows processed", unit="row") as pbar:
        for result in pool.imap_unordered(_process_chunk, reader):
            chunk_species, chunk_taxa, chunk_missing, processed = result
            
            # Merge results
            for (cell, week), species in chunk_species.items():
                global_species[(cell, week)].update(species)
            
            taxon_metadata.update(chunk_taxa)
            missing_cells.update(chunk_missing)
            
            total_processed += processed
            pbar.update(processed)

    pool.close()
    pool.join()

    logging.info(f"Finished processing {total_processed} rows")
    if missing_cells:
        logging.warning(f"Skipped {len(missing_cells)} H3 cells not present in environmental data")

    # Pivot collected species data into the H3 DataFrame
    logging.info("Building final dataset...")
    
    # Pre-generate week columns in the environmental dataframe
    num_rows = len(h3_df)
    for w in range(1, NUM_WEEKS + 1):
        h3_df[f'week_{w}'] = [[] for _ in range(num_rows)]

    # Map global_species dict into the dataframe
    h3_to_idx = {idx: i for i, idx in enumerate(h3_df['h3_index'].values)}
    for (cell, week), species_set in tqdm(global_species.items(), desc="Pivoting weeks"):
        if cell in h3_to_idx and 1 <= week <= NUM_WEEKS:
            row_idx = h3_to_idx[cell]
            h3_df.at[row_idx, f'week_{week}'] = list(species_set)

    # Convert to GeoDataFrame to ensure proper geo-metadata is ALWAYS saved
    logging.info(f"Saving combined dataset to {output_path}...")
    
    # If the h3_df was a standard pandas DataFrame, convert it back to a GeoDataFrame
    # Ensure geometry is in Shapely format, not WKB bytes
    if 'geometry' in h3_df.columns:
        from shapely import wkb
        if h3_df['geometry'].dtype == 'object' and len(h3_df) > 0 and isinstance(h3_df['geometry'].iloc[0], bytes):
            h3_df['geometry'] = h3_df['geometry'].apply(wkb.loads)
    
    # Explicitly create GeoDataFrame to write the GEOPARQUET metadata standard
    gdf = gpd.GeoDataFrame(h3_df, geometry='geometry' if 'geometry' in h3_df.columns else None)
    gdf.to_parquet(output_path, index=False)

    # Save taxonomy metadata for training label maps
    tax_path = output_path.replace('.parquet', '_taxonomy.csv')
    tax_df = pd.DataFrame.from_dict(taxon_metadata, orient='index')
    # Use species_code as the identifier column name to match the schema
    tax_df.index.name = 'species_code'
    tax_df.to_csv(tax_path)
    logging.info(f"Saved taxonomy metadata with {len(tax_df)} species to {tax_path}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description="Combine H3 and GBIF data.")
    parser.add_argument("--h3_path", required=True, help="Path to H3 environmental features parquet")
    parser.add_argument("--gbif_path", required=True, help="Path to processed GBIF CSV")
    parser.add_argument("--output_path", required=True, help="Output parquet path")
    parser.add_argument("--resolution", type=int, default=5, help="H3 resolution")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--classes", nargs="+", default=["Aves"], help="List of biological classes to include")
    parser.add_argument("--taxonomy_path", help="Path to master taxonomy.csv for label cleanup")

    args = parser.parse_args()
    combine_data(
        args.h3_path, 
        args.gbif_path, 
        args.output_path, 
        args.resolution, 
        args.workers, 
        args.classes,
        args.taxonomy_path
    )
