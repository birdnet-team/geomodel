import os
import argparse
from tqdm import tqdm
import zipfile
import pandas as pd
from dotenv import load_dotenv
import concurrent.futures
import math
import threading
import numpy as np
import json

# Load environment variables from .env file
load_dotenv()

# Working directory
WORKING_DIR = os.getenv('WORKING_DIRECTORY', '')

def estimate_zip_contents(file_path, csv_filename, sample_rows=10000):
    """
    Estimate the total number of rows in a zipped CSV file.
    
    Parameters:
    - file_path (str): The path to the zipped CSV file.
    - csv_filename (str): The name of the CSV file inside the zip archive.
    - sample_rows (int): Number of rows to sample for estimation.
    
    Returns:
    - tuple: (estimated_total_rows, estimated_total_size_bytes)
    """
    with zipfile.ZipFile(file_path, 'r') as z:
        info = z.getinfo(csv_filename)
        total_size_bytes = info.file_size
        
        # Read a sample of rows to estimate average row size
        with z.open(csv_filename) as f:
            # Skip header row
            header = f.readline()
            header_size = len(header)
            
            # Read sample rows
            sample_data = b''
            for _ in range(sample_rows):
                line = f.readline()
                if not line:
                    break
                sample_data += line
                
            # Calculate average row size
            if len(sample_data) > 0:
                avg_row_size = len(sample_data) / sample_rows
                # Estimate total rows (accounting for header)
                estimated_total_rows = (total_size_bytes - header_size) / avg_row_size
                return int(estimated_total_rows), total_size_bytes
            else:
                return 0, total_size_bytes

def estimate_gzip_rows(file_path, sample_rows=10000):
    """
    Estimate the total number of rows in a gzipped CSV file by sampling.
    
    Parameters:
    - file_path (str): Path to the gzipped CSV file
    - sample_rows (int): Number of rows to sample for estimation
    
    Returns:
    - int: Estimated total number of rows
    """
    import gzip
    
    # Get total file size
    total_size_bytes = os.path.getsize(file_path)
    
    # Read a sample to estimate average row size
    with gzip.open(file_path, 'rt') as f:
        # Skip header
        header = f.readline()
        header_size = len(header.encode())
        
        # Read sample rows
        sample_data = ''
        for _ in range(sample_rows):
            line = f.readline()
            if not line:
                break
            sample_data += line
            
    # Calculate average row size and estimate total rows
    if sample_data:
        avg_row_size = len(sample_data.encode()) / sample_rows
        estimated_total_rows = (total_size_bytes - header_size) / avg_row_size
        return max(int(estimated_total_rows), 1)  # Ensure at least 1 row
    else:
        return 0
    
def taxonomy_to_json():
    """
    Parse a taxonomy CSV file and convert it to a JSON dictionary mapping scientific names to species codes.
    """
    try:
        # Load the taxonomy data, ensuring all columns are read as strings
        tdf = pd.read_csv("taxonomy.csv", dtype=str)
        taxonomy_dict = {}
        
        # Track duplicate entries for logging
        duplicates = []
        
        for _, row in tdf.iterrows():
            # Get the primary name and code
            primary_name = row.get('sciName (Clements/eBird/ML)', '')
            
            # Get the code and check if it's NaN
            code = row.get('code (Clements/eBird/ML)', '')
            
            # Skip if code is NaN or empty
            if pd.isna(code) or code == '':
                continue
                
            # Now that we know code is valid, convert to string if needed
            code = str(code).strip()
            if not code:  # Skip rows with empty code after stripping
                continue
                
            # Process primary name
            if primary_name and not pd.isna(primary_name):
                primary_name = str(primary_name).strip()
                if primary_name:
                    if primary_name in taxonomy_dict and taxonomy_dict[primary_name] != code:
                        duplicates.append((primary_name, taxonomy_dict[primary_name], code))
                    taxonomy_dict[primary_name] = code
            
            # Process the synonyms
            try:
                synonyms_str = row.get('sciNameSynonyms', '[]')
                # Skip if synonyms is NaN
                if pd.isna(synonyms_str):
                    continue
                    
                # Process valid synonyms
                synonyms_str = str(synonyms_str)
                synonyms = synonyms_str.strip("[]").replace("'", "").replace('"', '').split(', ')
                
                # Add each synonym as a key
                for name in synonyms:
                    if not name or pd.isna(name):
                        continue
                        
                    name = str(name).strip()
                    if name:
                        if name in taxonomy_dict and taxonomy_dict[name] != code:
                            duplicates.append((name, taxonomy_dict[name], code))
                        taxonomy_dict[name] = code
            except Exception as e:
                print(f"Error processing synonyms for {primary_name}: {e}")
                
        # Report duplicates
        if duplicates:
            print(f"Warning: Found {len(duplicates)} duplicate entries in taxonomy file")
            
        # Save the dictionary to a JSON file
        with open("taxonomy.json", 'w') as f:
            import json
            json.dump(taxonomy_dict, f, indent=4)
            
        print(f"Saved {len(taxonomy_dict)} taxonomy entries to 'taxonomy.json'")
        return taxonomy_dict
    except Exception as e:
        print(f"Error in taxonomy_to_json: {e}")
        return {}

def sci_name_to_code(sci_name):
    """
    Convert a scientific name to its corresponding taxonomic code.
    
    Parameters:
    - sci_name (str): The scientific name to look up
    
    Returns:
    - str or None: The taxonomic code if found, None otherwise
    """
    # Initialize class variables if first call
    if not hasattr(sci_name_to_code, "taxonomy_dict"):
        sci_name_to_code.taxonomy_dict = None
        sci_name_to_code.lock = threading.Lock()
    
    # Skip processing for None or empty values
    if sci_name is None or not isinstance(sci_name, str) or not sci_name.strip():
        return None
    
    # Normalize input - trim whitespace and standardize capitalization
    sci_name = sci_name.strip()
    
    # Load taxonomy dictionary if not already loaded
    with sci_name_to_code.lock:
        if sci_name_to_code.taxonomy_dict is None:
            try:
                with open("taxonomy.json", 'r') as f:
                    sci_name_to_code.taxonomy_dict = json.load(f)
                print(f"Loaded taxonomy dictionary with {len(sci_name_to_code.taxonomy_dict):,} entries")
            except FileNotFoundError:
                print("Warning: taxonomy.json not found. Run taxonomy_to_json() first.")
                sci_name_to_code.taxonomy_dict = {}
            except json.JSONDecodeError:
                print("Warning: taxonomy.json is not valid JSON. Run taxonomy_to_json() again.")
                sci_name_to_code.taxonomy_dict = {}
            except Exception as e:
                print(f"Error loading taxonomy dictionary: {e}")
                sci_name_to_code.taxonomy_dict = {}
    
    # Look up the scientific name in the dictionary
    return sci_name_to_code.taxonomy_dict.get(sci_name)

def load_source_observations(file_path, csv_filename, chunk_size=100000, columns=None):
    """
    Load source observations from a zipped CSV file and yield DataFrames in chunks.
    
    Parameters:
    - file_path (str): The path to the zipped CSV file.
    - csv_filename (str): The name of the CSV file inside the zip archive.
    - chunk_size (int): The number of rows to read at a time.
    - columns (list): List of columns to keep. If None, keep all columns.
    
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
                
def process_inat_chunk(chunk):
    """
    Process a single chunk of iNaturalist data.
    """
    # Split occurenceID and make int
    chunk['occurrenceID'] = chunk['occurrenceID'].str.split('/').str[-1].astype(int, errors='ignore')
    
    # Round lat/lon to 3 decimal places
    chunk['decimalLatitude'] = chunk['decimalLatitude'].round(3)
    chunk['decimalLongitude'] = chunk['decimalLongitude'].round(3)
    
    # Split eventDate into date and time
    chunk[['eventDate', 'eventTime']] = chunk['eventDate'].str.split('T', expand=True)
    
    # Save time as hh:mm
    chunk['eventTime'] = chunk['eventTime'].str[:5]    
    
    # Convert column names to lowercase
    chunk.columns = [col.lower() for col in chunk.columns]
    
    return chunk

def parse_inat_source(threads=1):
    """
    Parse iNaturalist source observations and save them to a gzipped CSV file, using multithreading.
    
    Parameters:
    - threads (int): Number of threads to use for processing.
    
    Note: This can take a long time to run, we'll be processing it in chunks to avoid memory issues. The resulting file will be ~2.5GB.
    """    
    
    # Load iNaturalist observations
    # http://www.inaturalist.org/observations/gbif-observations-dwca.zip (Publication date May 27, 2025)
    # Citation: iNaturalist contributors, iNaturalist (2025). iNaturalist Research-grade Observations. iNaturalist.org. Occurrence dataset https://doi.org/10.15468/ab3s5x accessed via GBIF.org on 2025-06-02.
    inat_source_file = f"{WORKING_DIR}/gbif-observations-dwca.zip"
    inat_csv_filename = "observations.csv"
    inat_columns = ['occurrenceID', 'decimalLatitude', 'decimalLongitude', 'eventDate', 'taxonID', 'scientificName']
    inat_dst_file = f"{WORKING_DIR}/inat_parsed_observations.csv.gz"
    
    # Remove existing file if it exists
    if os.path.exists(inat_dst_file):
        os.remove(inat_dst_file)
    
    # Estimate the total number of rows and chunks
    chunk_size = 100000
    estimated_rows, file_size = estimate_zip_contents(inat_source_file, inat_csv_filename)
    estimated_chunks = math.ceil(estimated_rows / chunk_size)
    
    print(f"Processing iNaturalist data: ~{estimated_rows:,} rows ({file_size / (1024*1024*1024):.2f} GB)")
    print(f"Estimated number of chunks: {estimated_chunks:,} with {chunk_size:,} rows per chunk")
    
    # Initialize flags for header and counters
    write_header = True
    processed_chunks = 0
    processed_rows = 0
    
    # Create a generator for chunks
    chunk_generator = load_source_observations(inat_source_file, inat_csv_filename, chunk_size=chunk_size, columns=inat_columns)
    
    # Create global progress bar
    with tqdm(total=estimated_chunks, unit="chunk", desc="Overall iNaturalist progress") as global_pbar:
        # Process in batches
        while True:
            # Create a ThreadPoolExecutor for each batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                # Load next N chunks
                futures = []
                chunks_in_batch = 0
                
                # Try to load threads number of chunks
                for _ in range(threads):
                    try:
                        chunk = next(chunk_generator)
                        futures.append(executor.submit(process_inat_chunk, chunk))
                        chunks_in_batch += 1
                    except StopIteration:
                        # No more chunks to process
                        break
                
                # If no chunks were loaded, we're done
                if chunks_in_batch == 0:
                    break
                
                # Process and save the current batch
                with tqdm(total=chunks_in_batch, unit="cnk", desc=f"Batch {processed_chunks//threads + 1}/{math.ceil(estimated_chunks/threads)}") as batch_pbar:
                    for future in concurrent.futures.as_completed(futures):
                        chunk = future.result()
                        rows_in_chunk = len(chunk)
                        processed_rows += rows_in_chunk
                        
                        # Write header only for the first chunk
                        save_chunk_to_csv(chunk, inat_dst_file, mode='a', header=write_header and batch_pbar.n == 0)
                        write_header = False
                        batch_pbar.update(1)
                
                processed_chunks += chunks_in_batch
                global_pbar.update(chunks_in_batch)
                global_pbar.set_postfix(rows=f"{processed_rows:,}/{estimated_rows:,}", percentage=f"{processed_rows/estimated_rows*100:.1f}%")
    
    print(f"Processed {processed_chunks:,} chunks with {processed_rows:,} rows of iNaturalist data")

def process_ebird_chunk(chunk):
    """
    Process a single chunk of eBird data.
    """
    # Parse occurrenceid
    chunk['occurrenceid'] = chunk['occurrenceid'].str.split(':').str[-1].astype(int, errors='ignore')
    
    # Round lat/lon to 3 decimal places
    chunk['decimallatitude'] = chunk['decimallatitude'].round(3)
    chunk['decimallongitude'] = chunk['decimallongitude'].round(3)
    
    # Combine year, month, and day into a single date column (with leading zeros)
    chunk['year'] = chunk['year'].astype(str).str.zfill(4)
    chunk['month'] = chunk['month'].astype(str).str.zfill(2)
    chunk['day'] = chunk['day'].astype(str).str.zfill(2)
    chunk['eventdate'] = chunk[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1)
    chunk.drop(columns=['year', 'month', 'day'], inplace=True)
    
    # Avoid NaN values in individualcount
    chunk['individualcount'] = chunk['individualcount'].fillna(-1).astype(int)
    
    return chunk

def parse_ebird_source(threads=1):
    """
    Parse eBird source observations and save them to a gzipped CSV file, using multithreading.
    
    Parameters:
    - threads (int): Number of threads to use for processing.
    
    Note: This can take a long time to run, we'll be processing it in chunks to avoid memory issues. The resulting file will be ~27GB.
    """
    
    # Load eBird observations
    # https://hosted-datasets.gbif.org/eBird/2023-eBird-dwca-1.0.zip (Publication date March 22, 2024)
    # Citation: Auer T, Barker S, Barry J, Charnoky M, Curtis J, Davies I, Davis C, Downie I, Fink D, Fredericks T, Ganger J, Gerbracht J, Hanks C, Hochachka W, Iliff M, Imani J, Jordan A, Levatich T, Ligocki S, Long M T, Morris W, Morrow S, Oldham L, Padilla Obregon F, Robinson O, Rodewald A, Ruiz-Gutierrez V, Schloss M, Smith A, Smith J, Stillman A, Strimas-Mackey M, Sullivan B, Weber D, Wolf H, Wood C (2024). EOD – eBird Observation Dataset. Cornell Lab of Ornithology. Occurrence dataset https://doi.org/10.15468/aomfnb accessed via GBIF.org on 2025-06-02.
    ebird_source_file = f"{WORKING_DIR}/2023-eBird-dwca-1.0.zip"
    ebird_csv_filename = "eod.csv"
    ebird_columns = ['occurrenceid', 'decimallatitude', 'decimallongitude', 'year', 'month', 'day', 'scientificname', 'vernacularname', 'taxonconceptid', 'individualcount']
    ebird_dst_file = f"{WORKING_DIR}/ebird_parsed_observations.csv.gz"
    
    # Remove existing file if it exists
    if os.path.exists(ebird_dst_file):
        os.remove(ebird_dst_file)
    
    # Estimate the total number of rows and chunks
    chunk_size = 100000
    estimated_rows, file_size = estimate_zip_contents(ebird_source_file, ebird_csv_filename)
    estimated_chunks = math.ceil(estimated_rows / chunk_size)
    
    print(f"Processing eBird data: ~{estimated_rows:,} rows ({file_size / (1024*1024*1024):.2f} GB)")
    print(f"Estimated number of chunks: {estimated_chunks:,} with {chunk_size:,} rows per chunk")
    
    # Initialize flags for header and counters
    write_header = True
    processed_chunks = 0
    processed_rows = 0
    
    # Create a generator for chunks
    chunk_generator = load_source_observations(ebird_source_file, ebird_csv_filename, chunk_size=chunk_size, columns=ebird_columns)
    
    # Create global progress bar
    with tqdm(total=estimated_chunks, unit="chunk", desc="Overall eBird progress") as global_pbar:
        # Process in batches
        while True:
            # Create a ThreadPoolExecutor for each batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                # Load next N chunks
                futures = []
                chunks_in_batch = 0
                
                # Try to load threads number of chunks
                for _ in range(threads):
                    try:
                        chunk = next(chunk_generator)
                        futures.append(executor.submit(process_ebird_chunk, chunk))
                        chunks_in_batch += 1
                    except StopIteration:
                        # No more chunks to process
                        break
                
                # If no chunks were loaded, we're done
                if chunks_in_batch == 0:
                    break
                
                # Process and save the current batch
                with tqdm(total=chunks_in_batch, unit="cnk", desc=f"Batch {processed_chunks//threads + 1}/{math.ceil(estimated_chunks/threads)}") as batch_pbar:
                    for future in concurrent.futures.as_completed(futures):
                        chunk = future.result()
                        rows_in_chunk = len(chunk)
                        processed_rows += rows_in_chunk
                        
                        # Write header only for the first chunk
                        save_chunk_to_csv(chunk, ebird_dst_file, mode='a', header=write_header and batch_pbar.n == 0)
                        write_header = False
                        batch_pbar.update(1)
                
                processed_chunks += chunks_in_batch
                global_pbar.update(chunks_in_batch)
                global_pbar.set_postfix(rows=f"{processed_rows:,}/{estimated_rows:,}", percentage=f"{processed_rows/estimated_rows*100:.1f}%")
    
    print(f"Processed {processed_chunks:,} chunks with {processed_rows:,} rows of eBird data")
    
def get_week_from_date(date_series):
    """
    Get the week number from a date Series.
    
    Parameters:
    - date_series (pd.Series): A Series of datetime objects.
    
    Returns:
    - pd.Series: A Series of week numbers (1-48).
    """
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors='coerce')
    
    # Get the week number (1-48) (Normalized to 48 weeks)
    week_numbers = date_series.dt.isocalendar().week
    # Normalize to 48 weeks (1-48)
    week_numbers = ((week_numbers - 1) % 48) + 1        
    return week_numbers  
    
def merge_chunk_with_grid(chunk, grid_data, species_code_cache=None):
    """
    Merge a chunk of observation data with the grid data - optimized version.
    """
    # Initialize species code cache if not provided
    if species_code_cache is None:
        species_code_cache = {}
    
    # Ensure latitude and longitude are rounded to 3 decimal places
    chunk['decimallatitude'] = chunk['decimallatitude'].round(3)
    chunk['decimallongitude'] = chunk['decimallongitude'].round(3)
    
    # Get week number from eventDate - vectorized
    if 'eventdate' in chunk.columns:
        chunk['eventdate'] = pd.to_datetime(chunk['eventdate'], errors='coerce')
        chunk['week'] = get_week_from_date(chunk['eventdate'])
    else:
        print("eventdate column not found in chunk. Skipping this chunk.")
        return chunk
    
    # Vectorized distance calculation
    # Create arrays of all grid points
    grid_lats = grid_data.index.get_level_values('latitude').values
    grid_lons = grid_data.index.get_level_values('longitude').values
    
    # Create a dictionary to batch-update grid cells
    updates = {}
    
    # Process each row
    for _, row in chunk.iterrows():
        lat = row['decimallatitude']
        lon = row['decimallongitude']
        week = row['week']
        
        # Skip invalid coordinates or weeks
        if pd.isna(lat) or pd.isna(lon) or pd.isna(week):
            continue
            
        # Calculate distances to all grid points at once
        distances = ((grid_lats - lat)**2 + (grid_lons - lon)**2)**0.5
        
        # Find index of minimum distance
        if len(distances) == 0 or np.isnan(distances).all():
            continue
            
        min_idx = np.nanargmin(distances)
        closest_cell = (grid_lats[min_idx], grid_lons[min_idx])
        
        # Get species name
        species_name = row.get('scientificname', row.get('vernacularname', 'Unknown'))
        
        # Only genus and species, no subspecies
        if isinstance(species_name, str):
            species_name = ' '.join(species_name.split()[:2])
        else:
            continue
            
        # Use cached species code if available
        if species_name in species_code_cache:
            species_code = species_code_cache[species_name]
        else:
            species_code = sci_name_to_code(species_name)
            species_code_cache[species_name] = species_code  # Cache for future use
            
        if species_code is None:
            continue
        
        # Add to batch updates
        week_key = f'week_{int(week)}'
        if closest_cell not in updates:
            updates[closest_cell] = {}
            
        if week_key not in updates[closest_cell]:
            updates[closest_cell][week_key] = []
            
        updates[closest_cell][week_key].append(species_code)
    
    # Apply batch updates to grid_data
    for cell, week_updates in updates.items():
        for week_key, species_codes in week_updates.items():
            current = grid_data.at[cell, week_key]
            if isinstance(current, list):
                grid_data.at[cell, week_key] = current + species_codes
            else:
                grid_data.at[cell, week_key] = species_codes
    
    return chunk

def merge_with_grid(grid_file, threads=1):
    """
    Merge parsed observations with grid data.
    """
    # Observation files to merge with grid data
    parsed_ebird_file = f"{WORKING_DIR}/ebird_parsed_observations.csv.gz"
    parsed_inat_file = f"{WORKING_DIR}/inat_parsed_observations.csv.gz"    
    observation_files = [parsed_inat_file, parsed_ebird_file]
    
    if not os.path.exists(parsed_ebird_file) or not os.path.exists(parsed_inat_file):
        print("Parsed observation files not found. Please run the parsing scripts first.")
        return
    
    print(f"Merging parsed observations with grid file: {grid_file}")
    
    # Load grid data
    grid_data = pd.read_csv(grid_file)
    
    # Add week columns to grid data with empty lists
    for week in range(1, 49):
        grid_data[f'week_{week}'] = pd.Series([[] for _ in range(len(grid_data))], index=grid_data.index)
        
    # Ensure latitude and longitude are rounded to 3 decimal places
    grid_data['latitude'] = grid_data['latitude'].round(3)
    grid_data['longitude'] = grid_data['longitude'].round(3)
    grid_data.set_index(['latitude', 'longitude'], inplace=True)
    
    # Cache for species name to code lookups
    species_code_cache = {}
    
    # Process files in parallel batches
    for obs_file in observation_files:
        print(f"Processing observation file: {obs_file} with {threads} threads")
        
        # Estimate total rows
        estimated_rows = estimate_gzip_rows(obs_file)
        print(f"Estimated rows in {os.path.basename(obs_file)}: {estimated_rows:,}")
        
        # Load and process in chunks
        chunk_size = 50000  # Larger chunks for better vectorization
        processed_rows = 0
        
        with tqdm(total=estimated_rows, desc=f"Merging {os.path.basename(obs_file)}", unit="row") as pbar:
            # Use parallel processing for chunks if multiple threads
            if threads > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                    chunk_futures = []
                    
                    # Submit initial batch of chunks
                    for chunk in pd.read_csv(obs_file, chunksize=chunk_size):
                        if len(chunk_futures) >= threads * 2:  # Keep a buffer of 2x threads
                            # Wait for some to complete
                            done, not_done = concurrent.futures.wait(
                                chunk_futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            
                            # Process completed futures
                            for future in done:
                                pbar.update(len(future.result()))
                                processed_rows += len(future.result())
                            
                            # Update our list of futures to only include the ones still running
                            chunk_futures = list(not_done)
                                
                        # Submit new chunk
                        chunk_futures.append(
                            executor.submit(merge_chunk_with_grid, chunk, grid_data, species_code_cache)
                        )
                    
                    # Process remaining futures
                    for future in concurrent.futures.as_completed(chunk_futures):
                        pbar.update(len(future.result()))
                        processed_rows += len(future.result())
            else:
                # Sequential processing
                for chunk in pd.read_csv(obs_file, chunksize=chunk_size):
                    chunk = merge_chunk_with_grid(chunk, grid_data, species_code_cache)
                    processed_rows += len(chunk)
                    pbar.update(len(chunk))
                    
                    # DEBUG: Break after certain number of rows for testing
                    #if processed_rows >= 250000:
                    #    print("DEBUG: Stopping after 1,000,000 rows for testing")
                    #    break
    
    # Save the merged grid data to a CSV file
    save_path = f"{WORKING_DIR}/merged_observations_with_grid.csv.gz"
    print(f"Saving merged data to {save_path}...")
    grid_data.to_csv(save_path, compression='gzip')
    print(f"Saved merged data to {save_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parse observation data")
    parser.add_argument('--parse_inat_source', action='store_true', help="Parse iNaturalist source observations")
    parser.add_argument('--parse_ebird_source', action='store_true', help="Parse eBird source observations")
    parser.add_argument('--merge_with_grid', type=str, default='environmental_data_50km.csv', help="Provide path to a grid file to merge with parsed observations. Default is 'environmental_data_50km.csv' in the working directory.")
    parser.add_argument('--threads', type=int, default=8, help="Number of threads to use for processing (default: 8)")
    
    args = parser.parse_args()
    
    if args.parse_inat_source:
        parse_inat_source(threads=args.threads)
        
    if args.parse_ebird_source:
        parse_ebird_source(threads=args.threads)
        
    if args.merge_with_grid:
        grid_file = os.path.join(WORKING_DIR, args.merge_with_grid)
        if not os.path.exists(grid_file):
            print(f"Grid file '{grid_file}' does not exist. Please provide a valid path.")
        else:
            merge_with_grid(grid_file, threads=args.threads)