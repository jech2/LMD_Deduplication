import os
from .get_dataset_file_list import save_file_list, get_duplicated_query_list
from .get_lmd_clean_version_counts import save_version_count_meta
from .get_lmd_full_clean_match_file_list import get_matched_file_list
from .filepaths import *

if __name__ == '__main__':
    save_version_count_meta(LMD_CLEAN_DIR, '.mid', ret_file=LMD_CLEAN_DUP_DICT)
    
    save_file_list(LMD_CLEAN_DIR, '.mid', LMD_CLEAN_FILE_LIST)
    save_file_list(LMD_FULL_DIR, '.mid', LMD_FULL_FILE_LIST)
    
    get_duplicated_query_list(LMD_CLEAN_DUP_ALL_VERSIONS_DICT, LMD_CLEAN_DUP_FILE_LIST)
    get_matched_file_list(LMD_FULL_DIR, LMD_CLEAN_DIR, out_file=LMD_FULL_CLEAN_MATCH_FILE_LIST)