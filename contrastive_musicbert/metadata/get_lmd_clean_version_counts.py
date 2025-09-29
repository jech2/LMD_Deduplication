import os
import re
import shutil
from tqdm import tqdm
from collections import Counter
import json
import glob
import itertools
from pathlib import Path
from .utils import escape_special_chars, filepath_to_artist_and_title, filepath_to_artist_and_title_str 
from .filepaths import *

def find_files_with_different_versions(file_path):
    directory, filename = os.path.split(file_path)
    base_name, extension = os.path.splitext(filename)
    
    # 예: "Like A Virgin.1" -> "Like A Virgin"
    base_name_pattern = re.sub(r'\.\d+$', '', base_name)
    
    # add patterns without capitalization for finding files
    # escape special characters: () .
    case_insensitive_pattern = ''.join(['[{}{}]'.format(char.lower(), char.upper()) if char.isalpha() else escape_special_chars(char) for char in base_name_pattern])

    # add number patterns for finding versioned files
    versioned_pattern = f'{case_insensitive_pattern}.*\\{extension}'
    all_files = sorted(glob.glob(os.path.join(directory, '*')))
    
    # filter files with versions only
    versioned_files = [file for file in all_files if re.match(versioned_pattern, os.path.basename(file), re.IGNORECASE)]
    
    if len(versioned_files) == 0:
        # remove the numbering from the filename
        base_name_pattern = base_name.rsplit('.', 1)[0].lower()

        all_files = os.listdir(directory)
        
        # filter files with versions only
        versioned_files = [file for file in all_files if file.lower().startswith(base_name_pattern) and file.lower().endswith(extension.lower())]
    
    return versioned_files


def get_augmented_version_count_meta(json_data, ret_file='lmd_clean_version_counts_augmented.json'):
    new_dict = {}

    for key, value in json_data.items():
        for version in value["versions"]:
            new_dict[version] = value
            
    with open(ret_file, 'w') as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=4)
        
    print(f'Saved augmented metadata to {ret_file}')
    
def save_version_count_meta(source_dir, file_ext, ret_file='lmd_clean_version_counts.json'):
    version_count = {}
    file_meta = {} 

    duplicate_candidate_list = []

    # find files at least 2 versions
    for root, dirs, files in tqdm(os.walk(source_dir)):
        for file in files:
            if not file.endswith(f'{file_ext}'):
                continue
            
            if file.endswith(f'.1{file_ext}'):
                duplicate_candidate_list.append(os.path.join(root, file))

    for file in duplicate_candidate_list:
        original_file_pattern = file.replace(f'.1{file_ext}', f'{file_ext}')
        files_with_versions = find_files_with_different_versions(file)

        files_with_versions_str = [filepath_to_artist_and_title_str(file) for file in files_with_versions]

        assert len(files_with_versions) >= 1
                
        version_count[filepath_to_artist_and_title_str(file)] = {
            'count': len(files_with_versions),
            'versions': files_with_versions_str
        }
        
    print(f'Found {len(version_count)} files with versions. Saving metadata...')

    # save dict as json file
    with open(ret_file,'w') as f:
        json.dump(version_count, f, ensure_ascii=False, indent=4)
            
    print(f'Saved metadata to {ret_file}')
    
    get_augmented_version_count_meta(version_count, LMD_CLEAN_DUP_ALL_VERSIONS_DICT)
            
            
if __name__ == '__main__':
    save_version_count_meta(LMD_CLEAN_DIR, '.mid', ret_file=LMD_CLEAN_DUP_DICT)