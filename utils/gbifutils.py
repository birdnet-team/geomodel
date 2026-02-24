import logging
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

def date_to_week(day, month):
    """
    Converts a given day and month to a "birdNET" week number (1-48).
    """
    week = (month - 1) * 4
    if day <= 7:
        week += 1
    elif day <= 14:
        week += 2
    elif day <= 21:
        week += 3
    else:
        week += 4
    return int(week)

def estimate_rows(zip_archive, file_path, sample_rows=10000):
    """
    Estimate the total number of rows in a zipped CSV file by sampling.
    
    Parameters:
    - zip_archive (zipfile.ZipFile): Opened zip file object
    - file_path (str): Path to the CSV file within the zip archive
    - sample_rows (int): Number of rows to sample for estimation
    
    Returns:
    - int: Estimated total number of rows
    """
    # Get total file size
    total_size_bytes = zip_archive.getinfo(file_path).file_size
    
    # Read a sample to estimate average row size
    with zip_archive.open(file_path) as f:
        # Skip header
        header = f.readline()
        header_size = len(header.decode())
        
        # Read sample rows
        sample_data = ''
        for _ in range(sample_rows):
            line = f.readline()
            if not line:
                break
            sample_data += line.decode()
            
    # Calculate average row size and estimate total rows
    if sample_data:
        avg_row_size = len(sample_data.encode()) / sample_rows
        estimated_total_rows = (total_size_bytes - header_size) / avg_row_size
        return max(int(estimated_total_rows), 1)  # Ensure at least 1 row
    else:
        return 0

def date_to_week_vectorized(day, month):
    """
    Vectorized version of date_to_week. Converts day/month arrays to week numbers (1-48).
    """
    week = (month.astype(int) - 1) * 4
    week = week + np.where(day <= 7, 1, np.where(day <= 14, 2, np.where(day <= 21, 3, 4)))
    return week.astype(int)

def process_gbif_file(gbif_zip_path, file, output_csv_path, valid_classes=None, max_rows=None):

    required_cols = ['decimalLatitude', 'decimalLongitude', 'day', 'month', 'taxonKey', 'verbatimScientificName', 'class']

    output_df = pd.DataFrame({
        'latitude': [],
        'longitude': [],
        'taxon': [],
        'scientificName': [],
        'week': [],
        'class': []
    })

    output_df.to_csv(output_csv_path, index=False, encoding='utf-8')

    if valid_classes:
        valid_classes_set = set(valid_classes)

    rows_processed = 0

    with zipfile.ZipFile(gbif_zip_path, 'r') as z:
        estimated_rows = estimate_rows(z, file)
        with z.open(file) as f:
            with tqdm(total=estimated_rows, desc="Processing GBIF data") as pbar:
                for chunk in pd.read_csv(f, sep='\t', chunksize=100000, on_bad_lines='warn'):
                    chunk_size = len(chunk)

                    # Drop rows missing any required field
                    chunk = chunk.dropna(subset=required_cols)

                    if chunk.empty:
                        pbar.update(chunk_size)
                        rows_processed += chunk_size
                        if max_rows is not None and rows_processed >= max_rows:
                            break
                        continue

                    # Filter valid classes (vectorized)
                    if valid_classes:
                        chunk = chunk[chunk['class'].str.lower().isin(valid_classes_set)]

                    # Filter to full species only (exactly 1 or 2 words, ignoring subspecies and higher taxa)
                    chunk = chunk[chunk['verbatimScientificName'].str.split().str.len() <= 2]

                    if chunk.empty:
                        pbar.update(chunk_size)
                        rows_processed += chunk_size
                        if max_rows is not None and rows_processed >= max_rows:
                            break
                        continue

                    # Vectorized coordinate rounding
                    chunk = chunk.copy()
                    chunk['latitude'] = chunk['decimalLatitude'].astype(float).round(3)
                    chunk['longitude'] = chunk['decimalLongitude'].astype(float).round(3)

                    # Drop rows where coordinate conversion produced NaN
                    chunk = chunk.dropna(subset=['latitude', 'longitude'])

                    # Vectorized week computation
                    chunk['week'] = date_to_week_vectorized(
                        chunk['day'].astype(int),
                        chunk['month'].astype(int)
                    )

                    # Build output columns (taxonKey -> taxon position in CSV)
                    output_chunk = chunk[['latitude', 'longitude', 'taxonKey', 'verbatimScientificName', 'week', 'class']]
                    output_chunk.to_csv(output_csv_path, mode='a', header=False, index=False, encoding='utf-8')

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
    parser.add_argument('--valid_classes', nargs='*', default=['aves', 'amphibia', 'insecta', 'mammalia', 'reptilia'], help='List of classes to include (default: aves, amphibia, insecta, mammalia, reptilia)')
    parser.add_argument('--max_rows', type=int, default=None, help='Maximum number of rows to process (for testing purposes)')
    args = parser.parse_args()

    process_gbif_file(args.gbif, args.file, args.output, valid_classes=[cls.lower() for cls in args.valid_classes], max_rows=args.max_rows)