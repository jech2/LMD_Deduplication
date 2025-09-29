import os
import json

from pathlib import Path
from tqdm import tqdm
from .filepaths import *

def save_file_list(file_dir, ext, output_path=LMD_FULL_FILE_LIST):
    file_list = sorted(list(Path(file_dir).rglob(f'*{ext}')))

    with open(output_path, 'w') as f:
        for item in tqdm(file_list):
            p = Path(item)
            hash_dir = p.parent.name
            hash = p.stem
            f.write(f"{hash_dir}__{hash}\n")
    print('Done saving file list: ', file_dir, output_path)

def get_duplicated_query_list(vc_meta_path, output_path=LMD_CLEAN_DUP_FILE_LIST):
    with open(vc_meta_path) as f:
        json_data = json.load(f)

    # Write the selected filenames to the output file
    with open(output_path, 'w') as file:
        for key, value in json_data.items():
            file.write(f"{key}\n")
            
    print('Done saving duplicated query list: ', vc_meta_path, output_path)

    
if __name__ == '__main__':
    save_file_list(LMD_CLEAN_DIR, '.mid', LMD_CLEAN_FILE_LIST)
    save_file_list(LMD_FULL_DIR, '.mid', LMD_FULL_FILE_LIST)

    get_duplicated_query_list(LMD_CLEAN_DUP_ALL_VERSIONS_DICT, LMD_CLEAN_DUP_FILE_LIST)