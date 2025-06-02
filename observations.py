import os
import argparse
from tqdm import tqdm
import zipfile
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Working directory
WORKING_DIR = os.getenv('WORKING_DIRECTORY', '')

def load_source_observations(file_path, csv_filename, chunk_size=100000, columns=None):
    """
    Load source observations from a zipped CSV file and yield DataFrames in chunks.
    
    Parameters:
    - file_path (str): The path to the zipped CSV file.
    - chunk_size (int): The number of rows to read at a time.
    
    Yields:
    - pd.DataFrame: A DataFrame containing a chunk of the source observations.
    """
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open(csv_filename) as f:
            for chunk in pd.read_csv(f, chunksize=chunk_size):
                if columns is not None:
                    chunk = chunk[columns]
                yield chunk

def save_chunk_to_csv(chunk, output_file, mode='a', header=False):
    """
    Save a chunk of DataFrame to a gzipped CSV file.
    
    Parameters:
    - chunk (pd.DataFrame): The DataFrame chunk to save.
    - output_file (str): The path to the output gzipped CSV file.
    - mode (str): The file mode ('a' for append, 'w' for write).
    - header (bool): Whether to write the header row.
    """
    chunk.to_csv(output_file, mode=mode, index=False, header=header, compression='gzip')
                
def save_parsed_observations(df, output_file):
    """
    Save the DataFrame of parsed observations to a gzipped CSV file.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - output_file (str): The path to the output gzipped CSV file.
    """
    df.to_csv(output_file, index=False, compression='gzip')

def parse_inat_source():
    """
    Parse iNaturalist source observations and save them to a gzipped CSV file.
    
    Note: This can take a long time to run, dataset size is ~120GB and 110+ million rows. 
    We'll be processing it in chunks to avoid memory issues. The resulting file will be ~2.5GB.
    """    
    
    # Load iNaturalist observations
    # http://www.inaturalist.org/observations/gbif-observations-dwca.zip (Publication date May 27, 2025)
    # Citation: iNaturalist contributors, iNaturalist (2025). iNaturalist Research-grade Observations. iNaturalist.org. Occurrence dataset https://doi.org/10.15468/ab3s5x accessed via GBIF.org on 2025-06-02.
    inat_source_file = f"{WORKING_DIR}/gbif-observations-dwca.zip"
    inat_csv_filename = "observations.csv"
    inat_columns = ['occurrenceID', 'decimalLatitude', 'decimalLongitude', 'eventDate', 'taxonID', 'scientificName']
    inat_dst_file = f"{WORKING_DIR}/inat_parsed_observations.csv.gz"
    inat_df = pd.DataFrame(columns=inat_columns)
    
    for chunk in tqdm(load_source_observations(inat_source_file, inat_csv_filename, chunk_size=100000, columns=inat_columns), unit="cnk", desc="Loading iNaturalist observations"):
        
        # Split occurenceID and make int
        chunk['occurrenceID'] = chunk['occurrenceID'].str.split('/').str[-1].astype(int, errors='ignore')
        
        # Round lat/lon to 3 decimal places
        chunk['decimalLatitude'] = chunk['decimalLatitude'].round(3)
        chunk['decimalLongitude'] = chunk['decimalLongitude'].round(3)
        
        # Split eventDate into date and time
        chunk[['eventDate', 'eventTime']] = chunk['eventDate'].str.split('T', expand=True)
        
        # Save time as hh:mm
        chunk['eventTime'] = chunk['eventTime'].str[:5]      
            
        # Save chunk to gzipped CSV file
        save_chunk_to_csv(chunk, inat_dst_file, mode='a', header=inat_df.shape[0] == chunk.shape[0])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parse observation data")
    parser.add_argument('--parse_inat_source', action='store_true', help="Parse iNaturalist source observations")
    
    args = parser.parse_args()
    
    if args.parse_inat_source:
        parse_inat_source()