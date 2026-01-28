import logging
import math
import zipfile
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

def process_gbif_file(gbif_zip_path, file, output_csv_path):

    output_df = pd.DataFrame({
        'latitude': [],
        'longitude': [],
        'taxon': [],
        'week': []
    })

    output_df.to_csv(output_csv_path, index=False)

    with zipfile.ZipFile(gbif_zip_path, 'r') as z:
        estimated_rows = estimate_rows(z, file)
        with z.open(file) as f:
            with tqdm(total=estimated_rows, desc="Processing GBIF data") as pbar:
                for chunk in pd.read_csv(f, sep='\t', chunksize=10000, on_bad_lines='warn'):
                    # Collect rows in a list and create a DataFrame once per chunk
                    rows = []

                    # Process chunk
                    for idx, row in chunk.iterrows():
                        lat = row.get('decimalLatitude')
                        lon = row.get('decimalLongitude')
                        day = row.get('day')
                        month = row.get('month')
                        taxon = row.get('taxonKey')

                        if pd.isna(lat) or pd.isna(lon):  # Skip row if lat or lon is missing
                            continue
                        if pd.isna(day) or pd.isna(month):  # Skip row if day or month is missing
                            continue
                        if pd.isna(taxon):  # Skip row if taxon is missing
                            continue

                        # round coordinates to 3 decimal places
                        try:
                            latv = round(float(lat), 3)
                            lonv = round(float(lon), 3)
                        except Exception:
                            continue

                        week = date_to_week(int(day), int(month))
                        rows.append({'latitude': latv, 'longitude': lonv, 'taxon': taxon, 'week': week})

                    if rows:
                        output_chunk = pd.DataFrame.from_records(rows, columns=['latitude', 'longitude', 'taxon', 'week'])
                        output_chunk.to_csv(output_csv_path, mode='a', header=False, index=False)
                    pbar.update(len(chunk))

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process the zipped GBIF file and extract relevant columns.')
    parser.add_argument('--gbif', type=str, default="gbif_dev.zip", help='Path to the zipped GBIF file')
    parser.add_argument('--file', type=str, default="gbif_dev.csv", help='Name of the CSV file inside the zip')
    parser.add_argument('--output', type=str, default="./outputs/gbif_processed.gz", help='Output gzipped CSV file')
    args = parser.parse_args()

    gbif = args.gbif
    file = args.file
    output = args.output

    process_gbif_file(gbif, file, output)