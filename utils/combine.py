import geopandas as gpd
import gzip
import pandas as pd
import h3
import logging
import os

from tqdm import tqdm

def estimate_gzip_rows(file_path, sample_rows=10000):
    """
    Estimate the total number of rows in a gzipped CSV file by sampling.
    
    Parameters:
    - file_path (str): Path to the gzipped CSV file
    - sample_rows (int): Number of rows to sample for estimation
    
    Returns:
    - int: Estimated total number of rows
    """
    # Get compressed file size
    compressed_size = os.path.getsize(file_path)
    
    # Track compressed bytes read for the sample
    with gzip.open(file_path, 'rb') as f:
        # Read compressed header
        f.readline()  # Skip header in compressed form
        start_pos = f.fileobj.tell()
        
        # Read sample rows in compressed form
        for _ in range(sample_rows):
            line = f.readline()
            if not line:
                break
        
        end_pos = f.fileobj.tell()
    
    # Calculate compressed bytes used for sample
    compressed_sample_size = end_pos - start_pos
    
    if compressed_sample_size > 0:
        # Estimate based on compressed ratio
        estimated_total_rows = (compressed_size - start_pos) / (compressed_sample_size / sample_rows)
        return max(int(estimated_total_rows), 1)
    else:
        return 0

def combine_geodata_and_gbif(geodata_path, gbif_processed_path, output_path, valid_classes):
    """
    Combine geographical data with processed GBIF data to create a comprehensive dataset.
    
    Parameters:
    - geodata_path (str): Path to the parquet file containing geographical data
    - gbif_processed_path (str): Path to the gzipped CSV file with processed GBIF data
    - output_path (str): Path to save the combined parquet file
    - valid_classes (list): List of valid taxonomic classes to include from GBIF data
    """

    # Open the geopandas dataframe from a parquet file
    gdf = gpd.read_parquet(geodata_path)

    # Determine H3 resolution from the first cell
    first_row = gdf.iloc[0]
    cell = first_row['h3_index']
    h3_res = h3.get_resolution(cell)
    logging.info(f"Using H3 resolution {h3_res} based on first cell {cell}")

    # Initialize week columns
    for week in range(1, 49):
        gdf[f'week_{week}'] = [[] for _ in range(len(gdf))]

    # Estiamte rows for progressbar
    estimated_rows = estimate_gzip_rows(gbif_processed_path)

    with gzip.open(gbif_processed_path, 'rt', encoding='utf-8') as f:
        with tqdm(total=estimated_rows, desc="Processing GBIF data") as pbar:

            # iterate the gbif file in chunks and write observations to the correct h3 cell and week
            for chunk in pd.read_csv(f, chunksize=10000):
                for idx, row in chunk.iterrows():
                    taxon_class = row['class']

                    # Skip if class is not valid
                    if taxon_class not in valid_classes:
                        continue
                    lat = row['latitude']
                    lon = row['longitude']
                    taxon = row['taxon']

                    cell = h3.latlng_to_cell(lat, lon, h3_res)
                    week = int(row['week'])

                    gdf_row = gdf[gdf['h3_index'] == cell]
                    if not gdf_row.empty:
                        if taxon not in gdf_row[f'week_{week}']:
                            gdf.at[gdf_row.index[0], f'week_{week}'].append(taxon)
                    else:
                        logging.warning(f"Cell {cell} not found in geographical data.")

                pbar.update(len(chunk))


    gdf.to_parquet(output_path, index=False)


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Combine geographical data with processed GBIF data.')
    parser.add_argument('--geodata', type=str, default="./data/global_350km_ee.parquet", help='Path to the geographical data parquet file')
    parser.add_argument('--gbif', type=str, default="./outputs/gbif_processed.gz", help='Path to the gzipped processed GBIF file')
    parser.add_argument('--output', type=str, default=None, help='Output parquet file. If not provided will be generated based on geodata filename.')
    parser.add_argument('--valid_classes', type=str, nargs='+', default=['Aves', 'Mammalia', 'Amphibia'], help='List of valid taxonomic classes to include from GBIF data.')
    args = parser.parse_args()

    geodata = args.geodata
    gbif = args.gbif
    output = args.output    
    valid_classes = args.valid_classes
    if output is None:
        geodata_filename = os.path.basename(geodata).replace('.parquet', '_gbif.parquet')
        output = os.path.join('./outputs', geodata_filename)
        logging.info(f"No output file provided. Using default: {output}")


    combine_geodata_and_gbif(geodata, gbif, output, valid_classes)