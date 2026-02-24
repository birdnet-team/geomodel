import ast
import logging
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm


# Output CSV column order
OUTPUT_COLUMNS = ['latitude', 'longitude', 'taxonKey', 'verbatimScientificName', 'commonName', 'week', 'class']

# Required source columns (must all be non-null)
REQUIRED_COLUMNS = ['decimalLatitude', 'decimalLongitude', 'day', 'month', 'taxonKey', 'verbatimScientificName', 'class']


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
        valid_names: set of all valid scientific names (including synonyms)
        common_names: dict mapping sciName → common name (English)
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


def process_gbif_file(gbif_zip_path, file, output_csv_path, valid_classes=None, taxonomy_path=None, max_rows=None):
    """
    Process a GBIF Darwin Core Archive zip, filter and transform records,
    and write the result to a (optionally gzipped) CSV.

    If taxonomy_path is provided, only species listed in the taxonomy are kept.
    """
    # Write CSV header
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv_path, index=False, encoding='utf-8')

    valid_classes_set = set(valid_classes) if valid_classes else None
    if taxonomy_path:
        valid_species, common_names = load_taxonomy(taxonomy_path)
    else:
        valid_species, common_names = None, {}
    rows_processed = 0

    with zipfile.ZipFile(gbif_zip_path, 'r') as z:
        estimated_rows = estimate_rows(z, file)
        with z.open(file) as f:
            with tqdm(total=estimated_rows, desc="Processing GBIF data") as pbar:
                for chunk in pd.read_csv(f, sep='\t', chunksize=100000, on_bad_lines='warn',
                                        usecols=REQUIRED_COLUMNS):
                    chunk_size = len(chunk)

                    # Drop rows missing any required field
                    chunk = chunk.dropna(subset=REQUIRED_COLUMNS)

                    # Filter to valid taxonomic classes
                    if valid_classes_set and not chunk.empty:
                        chunk = chunk[chunk['class'].str.lower().isin(valid_classes_set)]

                    # Keep only full species (2 word names, skip subspecies / higher taxa)
                    if not chunk.empty:
                        chunk = chunk[chunk['verbatimScientificName'].str.split().str.len() == 2]

                    # Filter to species in taxonomy
                    if valid_species is not None and not chunk.empty:
                        chunk = chunk[chunk['verbatimScientificName'].isin(valid_species)]

                    if not chunk.empty:
                        chunk = chunk.copy()
                        chunk['latitude'] = chunk['decimalLatitude'].astype(float).round(3)
                        chunk['longitude'] = chunk['decimalLongitude'].astype(float).round(3)
                        chunk = chunk.dropna(subset=['latitude', 'longitude'])

                    if not chunk.empty:
                        chunk['week'] = date_to_week(chunk['day'], chunk['month'])
                        chunk['commonName'] = chunk['verbatimScientificName'].map(
                            lambda n: common_names.get(n, n)
                        )
                        chunk[OUTPUT_COLUMNS].to_csv(
                            output_csv_path, mode='a', header=False, index=False, encoding='utf-8'
                        )

                    pbar.update(chunk_size)
                    rows_processed += chunk_size
                    if max_rows is not None and rows_processed >= max_rows:
                        break


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
    args = parser.parse_args()

    process_gbif_file(
        args.gbif, args.file, args.output,
        valid_classes=[cls.lower() for cls in args.valid_classes],
        taxonomy_path=args.taxonomy,
        max_rows=args.max_rows,
    )