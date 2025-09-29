import glob
import json
import torch
import random
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from pytorch_lightning.loggers import WandbLogger
from datetime import datetime

from miditok import TokenizerConfig
from pathlib import Path
from symusic import Score
from miditok.pytorch_data import split_seq_in_subsequences
import argparse
from tqdm import tqdm

import pickle
from torch import nn
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
from hydra.core.hydra_config import HydraConfig

from contrastive_musicbert.data.custom_octuple import CustomOctuple
from contrastive_musicbert.utils.general_utils import set_manual_seed
from contrastive_musicbert.utils.metadata_handler import MetadataHandler

from contrastive_musicbert.utils.inference_utils import get_musicbert_last_hidden_embedding

from contrastive_musicbert.data.data import DataModule
from contrastive_musicbert.model.BERT import BERT_Lightning
from contrastive_musicbert.metadata.filepaths import LMD_CLEAN_DIR, LMD_FULL_DIR

@hydra.main(version_base=None, config_path="./yamls/", config_name="inference")
def main(cfg: DictConfig):
    if cfg.checkpoint_folder is None:
        raise ValueError("checkpoint_folder is None")
    if cfg.checkpoint_file is None:
        raise ValueError("checkpoint_file is None")
    if cfg.file_ext is None:
        raise ValueError("file_ext is None")
    if cfg.embedding_mode is None:
        raise ValueError("embedding_mode is None")
    
    config_path = Path(cfg.checkpoint_folder) / '.hydra' / 'config.yaml'
    config = OmegaConf.load(config_path)
    print('load config from ', config_path)
    
    vocab = CustomOctuple()    

    random_seed = config.random_seed
    set_manual_seed(random_seed)

    torch.set_float32_matmul_precision("high")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('use ', device)

    model = BERT_Lightning(
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        dim_head=int(config.dim / config.heads),
        mlp_dim=int(4 * config.dim),
        max_len=config.max_len,
        rate=config.rate,
        loss_weights=config.loss_weights,
        lr=config.lr,
        warm_up=config.warm_up,
        temp=config.temp,
        mode=config.mode,
        vocab=vocab,
    ).to(device)

    checkpoint_folder = Path(cfg.checkpoint_folder)
    checkpoint_file = cfg.checkpoint_file
    file_ext = cfg.file_ext 
        
    checkpoint_path = checkpoint_folder / checkpoint_file

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    metadata_handler = MetadataHandler()
    if cfg.infer_dataset == 'lmd_clean':
        file_list = metadata_handler.get_lmd_clean_file_list()
        infer_dataset_dir = LMD_CLEAN_DIR
    elif cfg.infer_dataset == 'lmd_full':
        file_list = metadata_handler.get_lmd_full_file_list()
        infer_dataset_dir = LMD_FULL_DIR
    else:
        NotImplementedError('no inference dataset selected: available lmd-full and lmd-clean')

    embed_dict = dict()

    for sample_song in tqdm(file_list):
        try:
            sample_midi_path = Path(
                metadata_handler.get_artist_and_title_str_to_filepath(sample_song, ref_dir=infer_dataset_dir)
            )
            save_path = sample_midi_path.with_name(sample_midi_path.stem + file_ext)
            
            if save_path.exists():
                print('already exists', save_path)
                continue
            avged_emb = get_musicbert_last_hidden_embedding(model, vocab, sample_midi_path, embedding_mode=cfg.embedding_mode) # output h is already the h[:, 0] (CLS token)
            
            np.save(save_path, avged_emb.data.detach().cpu())
            print('saved', save_path)

        except Exception as e:
            error_log = checkpoint_folder / "inference_error_log.txt"
            with error_log.open("a+", encoding="utf-8") as f:
                print(f'Error in {sample_midi_path} : {e}')
                f.write(f'Error in {sample_midi_path} : {e}\n')
        
            

if __name__ == "__main__":
    main()