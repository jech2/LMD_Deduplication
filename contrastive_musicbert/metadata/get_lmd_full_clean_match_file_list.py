import os
import hashlib
from multiprocessing import Pool
from tqdm import tqdm
from .utils import filepath_to_artist_and_title_str
from .filepaths import *

def get_file_hash(file_path):
    hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash.hexdigest()

def compare_files(args):
    query_file, target_files_dict = args
    query_hash = get_file_hash(query_file)
    if query_hash in target_files_dict:
        return (filepath_to_artist_and_title_str(query_file), filepath_to_artist_and_title_str(target_files_dict[query_hash]))
    return None

def get_matched_file_list(lmd_full_dir, lmd_clean_dir, out_file=LMD_FULL_CLEAN_MATCH_FILE_LIST):
    cleaned_midi_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(lmd_clean_dir) for f in filenames if f.endswith('.mid')]
    lmd_full_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(lmd_full_dir) for f in filenames if f.endswith('.mid')]

    print(f"Found {len(cleaned_midi_files)} cleaned MIDI files")
    print(f"Found {len(lmd_full_files)} LMD Full MIDI files")

    # Create a dictionary with target files hashes  
    target_files_dict = {get_file_hash(file): file for file in tqdm(lmd_full_files, desc="Generating file hash target dict")}

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(compare_files, [(query_file, target_files_dict) for query_file in cleaned_midi_files]), total=len(cleaned_midi_files)))

    # Filter out None results
    matched_files = [result for result in results if result is not None]
    

    # Save to CSV
    import csv
    with open(out_file, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(['Query File', 'Target File'])
        for match in matched_files:
            
            filewriter.writerow(match)

if __name__ == "__main__":
    get_matched_file_list(LMD_FULL_DIR, LMD_CLEAN_DIR, out_file=LMD_FULL_CLEAN_MATCH_FILE_LIST)
