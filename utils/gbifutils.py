"""GBIF data processing utilities.

Functions for reading, filtering, and transforming GBIF Darwin Core Archive
records into a clean CSV suitable for downstream H3 aggregation.
"""

import ast
import gzip
import io
import logging
import multiprocessing as mp
import os
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm


# Output CSV column order
OUTPUT_COLUMNS = ['latitude', 'longitude', 'taxonKey', 'verbatimScientificName', 'commonName', 'week', 'class']

# Required source columns (must all be non-null)
REQUIRED_COLUMNS = ['decimalLatitude', 'decimalLongitude', 'day', 'month', 'taxonKey', 'verbatimScientificName', 'class']

# Block size for parallel reading (64 MB of uncompressed text per block)
_BLOCK_SIZE = 64 * 1024 * 1024


def date_to_week(day, month):
    """
    Convert day/month arrays to BirdNET week numbers (1-48, 4 weeks per month).
    Works with both scalar and vectorized (numpy array) inputs.
    """
    week = (np.asarray(month, dtype=int) - 1) * 4
    day = np.asarray(day, dtype=int)
    week = week + np.where(day <= 7, 1, np.where(day <= 14, 2, np.where(day <= 21, 3, 4)))
    return week


def estimate_rows(zip_archive, file_path, sample_rows=10000):
    """
    Estimate the total number of rows in a zipped CSV file by sampling.
    """
    total_size_bytes = zip_archive.getinfo(file_path).file_size

    with zip_archive.open(file_path) as f:
        header = f.readline()
        header_size = len(header.decode())

        sample_data = b''
        for _ in range(sample_rows):
            line = f.readline()
            if not line:
                break
            sample_data += line

    if sample_data:
        avg_row_size = len(sample_data) / sample_rows
        estimated_total_rows = (total_size_bytes - header_size) / avg_row_size
        return max(int(estimated_total_rows), 1)
    return 0


def load_taxonomy(taxonomy_path):
    """
    Load taxonomy CSV.

    Returns:
        valid_names (set): All valid scientific names (including synonyms).
        common_names (dict): Mapping of sciName to common name (English).
    """
    df = pd.read_csv(taxonomy_path)
    valid_names = set()
    common_names: dict = {}

    # Build sciName → commonName lookup from primary name column
    common_col = 'comNameEn (Clements/eBird/ML)'
    fallback_col = 'comNameEn (IOC)'

    # Collect names from synonym lists (covers primary names too)
    # and map each synonym to its row's common name
    if 'sciNameSynonyms' in df.columns:
        for _, row in df.iterrows():
            val = row.get('sciNameSynonyms')
            if pd.isna(val):
                continue
            try:
                names = ast.literal_eval(str(val))
                if isinstance(names, list):
                    valid_names.update(names)
                    # Resolve common name: prefer Clements, fallback IOC, else sci name
                    com = row.get(common_col)
                    if pd.isna(com) or not str(com).strip():
                        com = row.get(fallback_col)
                    if pd.isna(com) or not str(com).strip():
                        com = row.get('sciName (Clements/eBird/ML)', '')
                    com = str(com).strip() if not pd.isna(com) else ''
                    for n in names:
                        if n and n not in common_names:
                            common_names[n] = com if com else n
            except (ValueError, SyntaxError):
                pass

    # Also add explicit sci name columns as fallback
    for col in ['sciName (Clements/eBird/ML)', 'sciName (GBIF)', 'sciName (IOC)']:
        if col in df.columns:
            valid_names.update(df[col].dropna().astype(str).values)

    valid_names.discard('')
    logging.info(f"Loaded {len(valid_names)} valid species names from taxonomy")
    return valid_names, common_names


def _read_blocks(stream, block_size):
    """Yield byte blocks from *stream*, each ending on a newline boundary."""
    remainder = b''
    while True:
        raw = stream.read(block_size)
        if not raw:
            if remainder:
                yield remainder
            break
        data = remainder + raw
        last_nl = data.rfind(b'\n')
        if last_nl == -1:
            remainder = data
            continue
        yield data[:last_nl + 1]
        remainder = data[last_nl + 1:]


# ---------------------------------------------------------------------------
# Worker pool helpers  (module-level for pickling)
# ---------------------------------------------------------------------------

_wctx: dict = {}


def _init_worker(header_bytes, valid_species_list, valid_classes_list, common_names_dict):
    """Populate per-worker read-only state (called once per pool process)."""
    _wctx['header'] = header_bytes
    _wctx['valid_species'] = set(valid_species_list) if valid_species_list else None
    _wctx['valid_classes'] = set(valid_classes_list) if valid_classes_list else None
    _wctx['common_names'] = dict(common_names_dict) if common_names_dict else {}


def _filter_block(block_bytes):
    """Parse a raw TSV byte block, filter rows, return (csv_str, n_rows, block_len)."""
    header = _wctx['header']
    valid_species = _wctx['valid_species']
    valid_classes_set = _wctx['valid_classes']
    common_names = _wctx['common_names']

    block_len = len(block_bytes)

    data = header + b'\n' + block_bytes
    try:
        chunk = pd.read_csv(io.BytesIO(data), sep='\t', dtype=str,
                            usecols=REQUIRED_COLUMNS, on_bad_lines='skip')
    except Exception:
        return '', 0, block_len

    n_rows = len(chunk)

    chunk = chunk.dropna(subset=REQUIRED_COLUMNS)

    # Filter to species in taxonomy (most selective — do first)
    if valid_species is not None and not chunk.empty:
        chunk = chunk[chunk['verbatimScientificName'].isin(valid_species)]

    # Filter to valid taxonomic classes
    if valid_classes_set and not chunk.empty:
        chunk = chunk[chunk['class'].str.lower().isin(valid_classes_set)]

    # Keep only full species (exactly one space → binomial name)
    if not chunk.empty:
        chunk = chunk[chunk['verbatimScientificName'].str.count(' ') == 1]

    if not chunk.empty:
        lat = pd.to_numeric(chunk['decimalLatitude'], errors='coerce')
        lon = pd.to_numeric(chunk['decimalLongitude'], errors='coerce')
        valid_coords = lat.notna() & lon.notna()
        chunk = chunk.loc[valid_coords].copy()
        chunk['latitude'] = lat[valid_coords].round(3)
        chunk['longitude'] = lon[valid_coords].round(3)

    if not chunk.empty:
        chunk['week'] = date_to_week(chunk['day'], chunk['month'])
        chunk['commonName'] = chunk['verbatimScientificName'].map(
            lambda n: common_names.get(n, n)
        )
        return chunk[OUTPUT_COLUMNS].to_csv(index=False, header=False), n_rows, block_len

    return '', n_rows, block_len


# ---------------------------------------------------------------------------
# Main processing entry point
# ---------------------------------------------------------------------------

def process_gbif_file(gbif_zip_path, file, output_csv_path, valid_classes=None,
                      taxonomy_path=None, max_rows=None, n_workers=None):
    """Process a GBIF Darwin Core Archive zip using parallel workers.

    Reads raw byte blocks from the zip sequentially, then distributes
    parsing and filtering across *n_workers* processes.

    Args:
        gbif_zip_path (str): Path to the GBIF Darwin Core Archive zip file.
        file (str): Name of the CSV/TSV file inside the zip.
        output_csv_path (str): Output path for the processed CSV.
        valid_classes (list[str] | None): List of taxonomic classes to keep.
        taxonomy_path (str | None): Path to taxonomy CSV for filtering.
        max_rows (int | None): Maximum number of rows to process.
        n_workers (int | None): Number of parallel worker processes.
            Default: ``min(cpu_count - 1, 8)``.
    """
    if n_workers is None:
        n_workers = min(max(1, os.cpu_count() - 1), 8)

    valid_classes_list = [c.lower() for c in valid_classes] if valid_classes else None
    if taxonomy_path:
        valid_species, common_names = load_taxonomy(taxonomy_path)
    else:
        valid_species, common_names = None, {}

    use_gzip = str(output_csv_path).endswith('.gz')

    with zipfile.ZipFile(gbif_zip_path, 'r') as z:
        file_size = z.getinfo(file).file_size

        with z.open(file) as f:
            header_line = f.readline().rstrip(b'\n\r')

            # Open output stream
            if use_gzip:
                out = gzip.open(output_csv_path, 'wt', encoding='utf-8', compresslevel=6)
            else:
                out = open(output_csv_path, 'w', encoding='utf-8')
            out.write(','.join(OUTPUT_COLUMNS) + '\n')

            init_args = (
                header_line,
                list(valid_species) if valid_species else None,
                valid_classes_list,
                common_names,
            )

            total_rows = 0
            rows_written = 0

            try:
                with mp.Pool(n_workers, initializer=_init_worker,
                             initargs=init_args) as pool:
                    blocks = _read_blocks(f, _BLOCK_SIZE)
                    results = pool.imap(_filter_block, blocks, chunksize=1)

                    with tqdm(total=file_size,
                              desc=f"Processing GBIF ({n_workers} workers)",
                              unit='B', unit_scale=True) as pbar:
                        for csv_str, n_rows, block_len in results:
                            if csv_str:
                                out.write(csv_str)
                                rows_written += csv_str.count('\n')
                            total_rows += n_rows
                            pbar.update(block_len)

                            if max_rows and total_rows >= max_rows:
                                pool.terminate()
                                break
            finally:
                out.close()

    logging.info(f"Processed {total_rows:,} rows, wrote {rows_written:,} records "
                 f"to {output_csv_path}")


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process the zipped GBIF file and extract relevant columns.')
    parser.add_argument('--gbif', type=str, default="gbif_dev.zip", help='Path to the zipped GBIF file')
    parser.add_argument('--file', type=str, default="gbif_dev.csv", help='Name of the CSV file inside the zip')
    parser.add_argument('--output', type=str, default="./outputs/gbif_processed.gz", help='Output gzipped CSV file')
    parser.add_argument('--valid_classes', nargs='*', default=['aves', 'amphibia', 'insecta', 'mammalia', 'reptilia'],
                        help='List of classes to include (default: aves, amphibia, insecta, mammalia, reptilia)')
    parser.add_argument('--taxonomy', type=str, default=None,
                        help='Path to taxonomy CSV — only species in the taxonomy are kept')
    parser.add_argument('--max_rows', type=int, default=None, help='Maximum number of rows to process (for testing)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: min(cpu_count-1, 8))')
    args = parser.parse_args()

    process_gbif_file(
        args.gbif, args.file, args.output,
        valid_classes=[cls.lower() for cls in args.valid_classes],
        taxonomy_path=args.taxonomy,
        max_rows=args.max_rows,
        n_workers=args.workers,
    )