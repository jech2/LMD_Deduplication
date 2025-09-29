import json
import csv
from ..metadata.filepaths import *
from ..metadata.utils import (
    filepath_to_artist_and_title_str,
    artist_and_title_str_to_filepath,
)

class MetadataHandler:
    def __init__(self, use_dir=None, ref_ext=".mid"):

        self.lmd_full_file_list = None
        self.lmd_clean_file_list = None
        self.lmd_clean_dup_file_list = None
        self.lmd_clean_match_file_list = None
        self.lmd_full_match_file_list = None
        self.lmd_clean_dup_dict = None
        self.lmd_clean_dup_all_versions_dict = None

        self.use_dir = use_dir
        self.ref_ext = ref_ext
        self.load_metadata(use_dir, ref_ext)

    def load_metadata(self, use_dir='ref', ref_ext=".mid"):
        if use_dir:
            if use_dir == 'ref':
                ref_dir_A = LMD_FULL_DIR
                ref_dir_B = LMD_CLEAN_DIR
            else:
                ref_dir_A = use_dir.split('__')[0]
                ref_dir_B = use_dir.split('__')[1]
                if ref_dir_A == '':
                    ref_dir_A = ref_dir_B
            print('ref_dir_A: ', ref_dir_A)
            print('ref_dir_B: ', ref_dir_B)
            self.lmd_full_file_list = self.load_file_list(
                LMD_FULL_FILE_LIST, ref_dir=ref_dir_A, ref_ext=ref_ext
            )
            self.lmd_clean_file_list = self.load_file_list(
                LMD_CLEAN_FILE_LIST, ref_dir=ref_dir_B, ref_ext=ref_ext
            )
            self.lmd_clean_dup_file_list = self.load_file_list(
                LMD_CLEAN_DUP_FILE_LIST, ref_dir=ref_dir_B, ref_ext=ref_ext
            )
            self.lmd_clean_match_file_list, self.lmd_full_match_file_list = (
                self.load_match_csv(
                    LMD_FULL_CLEAN_MATCH_FILE_LIST,
                    lmd_full_dir=ref_dir_A,
                    lmd_clean_dir=ref_dir_B,
                    ref_ext=ref_ext,
                )
            )
            self.lmd_clean_dup_dict = self.load_dict(
                LMD_CLEAN_DUP_DICT, ref_dir=ref_dir_B, ref_ext=ref_ext
            )
            self.lmd_clean_dup_all_versions_dict = self.load_dict(
                LMD_CLEAN_DUP_ALL_VERSIONS_DICT, ref_dir=ref_dir_B, ref_ext=ref_ext
            )
        else:
            self.lmd_full_file_list = self.load_file_list(LMD_FULL_FILE_LIST)
            self.lmd_clean_file_list = self.load_file_list(LMD_CLEAN_FILE_LIST)
            self.lmd_clean_dup_file_list = self.load_file_list(LMD_CLEAN_DUP_FILE_LIST)
            self.lmd_clean_match_file_list, self.lmd_full_match_file_list = (
                self.load_match_csv(LMD_FULL_CLEAN_MATCH_FILE_LIST)
            )
            self.lmd_clean_dup_dict = self.load_dict(LMD_CLEAN_DUP_DICT)
            self.lmd_clean_dup_all_versions_dict = self.load_dict(
                LMD_CLEAN_DUP_ALL_VERSIONS_DICT
            )

    def load_file_list(self, file_path, ref_dir=None, ref_ext=".mid"):
        with open(file_path, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        if ref_dir is not None:
            lines = [
                artist_and_title_str_to_filepath(line, ref_dir, ref_ext)
                for line in lines
            ]
        return lines

    def load_dict(self, dict_path, ref_dir=None, ref_ext=".mid"):
        with open(dict_path, "r") as f:
            js = json.load(f)
        if ref_dir is not None:
            new_js = {}
            for key, value in js.items():
                new_key = artist_and_title_str_to_filepath(key, ref_dir, ref_ext)
                value["versions"] = [
                    artist_and_title_str_to_filepath(v, ref_dir, ref_ext)
                    for v in value["versions"]
                ]  # update version paths
                new_js[new_key] = value
            js = new_js

        return js

    def load_match_csv(
        self, csv_path, lmd_full_dir=None, lmd_clean_dir=None, ref_ext=".mid"
    ):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            lmd_clean_matches = list(reader)

        lmd_clean_match = [
            i[0] for i in lmd_clean_matches
        ]  # col_idx=0 means lmd-clean's file, col_idx=1 means lmd-full's file
        lmd_full_match = [i[1] for i in lmd_clean_matches]

        if lmd_full_dir is not None:
            lmd_full_match = [
                artist_and_title_str_to_filepath(i, lmd_full_dir, ref_ext)
                for i in lmd_full_match
            ]
        if lmd_clean_dir is not None:
            lmd_clean_match = [
                artist_and_title_str_to_filepath(i, lmd_clean_dir, ref_ext)
                for i in lmd_clean_match
            ]

        return lmd_clean_match, lmd_full_match

    def get_lmd_full_file_list(self):
        return self.lmd_full_file_list

    def get_lmd_clean_file_list(self):
        return self.lmd_clean_file_list

    def get_lmd_clean_dup_file_list(self):
        return self.lmd_clean_dup_file_list

    def get_lmd_full_match_file_list(self):
        return self.lmd_full_match_file_list

    def get_lmd_clean_match_file_list(self):
        return self.lmd_clean_match_file_list

    def get_all_version_count_meta(self):
        return self.lmd_clean_dup_all_versions_dict

    def get_filepath_to_artist_and_title_str(self, filepath):
        return filepath_to_artist_and_title_str(filepath)

    def get_artist_and_title_str_to_filepath(
        self, artist_and_title_str, ref_dir=LMD_CLEAN_DIR, ref_ext='.mid'
    ):
        return artist_and_title_str_to_filepath(artist_and_title_str, ref_dir, ref_ext)

    def get_file_list(self, file_name):
        if file_name == LMD_CLEAN_FILE_LIST:
            return self.get_lmd_clean_file_list()
        elif file_name == LMD_FULL_FILE_LIST:
            return self.get_lmd_full_file_list()
        elif file_name == LMD_CLEAN_DUP_FILE_LIST:
            return self.get_lmd_clean_dup_file_list()
        else:
            raise ValueError("No such file list available")

    def get_ref_dir(self, file_name):
        if file_name == LMD_CLEAN_FILE_LIST or file_name == 'lmd_clean':
            return LMD_CLEAN_DIR
        elif file_name == LMD_FULL_FILE_LIST or file_name == 'lmd_full':
            return LMD_FULL_DIR
        elif file_name == LMD_CLEAN_DUP_FILE_LIST or file_name == 'lmd_dup':
            return LMD_CLEAN_DIR
        else:
            raise ValueError("No such file list available")


if __name__ == "__main__":
    mh = MetadataHandler()
    print(mh.get_lmd_full_match_file_list())
    print(mh.get_lmd_clean_match_file_list())
    print(mh.get_all_version_count_meta())
