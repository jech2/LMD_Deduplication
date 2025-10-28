import os
import pickle
import argparse
import multiprocessing
import csv
from time import time
from tqdm import tqdm

from pathlib import Path
from symusic import Score

from .custom_octuple import CustomOctuple
import numpy as np
import json

from ..utils.metadata_handler import MetadataHandler

def convert(file_path):
    try:
        midi = Score(file_path)
        tokens = tokenizer(midi)
        converted_back_midi = tokenizer(tokens)  

        file_name = file_path.stem
        parent_folder = file_path.parent.stem
        dest_folder_path = dest_path / parent_folder

        if not dest_folder_path.exists():
            dest_folder_path.mkdir(parents=True)
        
        n_tokens = [len(tokenizer._vocab_base[i]) for i in range(len(tokenizer._vocab_base))]

        tokens_arr = np.array(tokens)
        for i in range(len(n_tokens)):
            assert max(tokens_arr[:, i]) < n_tokens[i], f"{i} {max(tokens_arr[:, i])} {n_tokens[i]}"

        tokenizer.save_tokens(tokens, dest_folder_path / (file_name + ".json"))
        # converted_back_midi.dump_midi(dest_folder_path / (file_name + ".mid"))
        
    except Exception as error:
        print("File error occured!", error, file_path)
        with open(dest_path / "conversion_error_files.txt", "a+") as f:
            f.write(str(file_path) + "\n")
        

if __name__ == "__main__":
    global dest_path
    global tokenizer
    global drop_drum
    global conversion_error_count

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="./data/lmd_full/")
    parser.add_argument("--dest_path", type=str, default="./data/lmd_filt_octuple/")
    parser.add_argument("--num_process", type=int, default=10)
    parser.add_argument("--filter_with_lmd_clean", action="store_true")
    
    args = parser.parse_args()
    dest_path = Path(args.dest_path)

    tokenizer = CustomOctuple() 
    n_tokens = [len(tokenizer._vocab_base[i]) for i in range(len(tokenizer._vocab_base))]
    print(n_tokens, 'tokens of Octuple preprocessing...')
    
    start_time = time()
    if args.filter_with_lmd_clean:
        metadata_handler = MetadataHandler()
        lmd_clean_match = metadata_handler.get_lmd_full_match_file_list()
        print('Filter LMD_clean matched files')
    else:
        lmd_clean_match = []
        print('Not filter LMD_clean matched files')

    file_list = list(Path(args.src_path).glob("*/*.mid"))
    file_list = [str(i) for i in file_list]
    print('file_list example: ', file_list[0])
    
    # remove files that are matching with lmd_clean_match
    print(f'Remove {len(lmd_clean_match)}/{len(file_list)} files matched via MD5 file hash matching...')



    lmd_clean_match = set(lmd_clean_match)
    file_list = [item for item in file_list if item not in lmd_clean_match]

    file_list = [i for i in file_list if 'd39f4e1f2ac8e56aa2f6b986a2609546' in i]

    file_list = [Path(i) for i in file_list]

    print(f'{len(file_list)} files are processed...')

    pool = multiprocessing.Pool(args.num_process)
    for _ in tqdm(pool.imap_unordered(convert, file_list), total=len(file_list)):
        pass
    pool.close()
    pool.join()

    print(f"Time : {time() - start_time:.3f} sec")