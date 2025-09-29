import os
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
from contrastive_musicbert.metadata.filepaths import LMD_CLEAN_DIR, LMD_FULL_DIR

def get_debug_inputs(ref_query_file):
    metadata_handler = MetadataHandler()
    all_version_count_meta = metadata_handler.get_all_version_count_meta()
    dup_files = all_version_count_meta[ref_query_file]['versions']
    
    return ref_query_file, dup_files

def debug_inference(model, vocab, ref_query="Van Halen__Jump"):
    ref_file = MetadataHandler().get_artist_and_title_str_to_filepath(ref_query, LMD_CLEAN_DIR, '.mid')
    _, dup_files = get_debug_inputs(ref_query)
    ref_emb = get_musicbert_last_hidden_embedding(model, vocab, ref_file)
    avged_embs = []    

    for file in tqdm(dup_files):
        file = MetadataHandler().get_artist_and_title_str_to_filepath(file, LMD_CLEAN_DIR, '.mid')
        avged_emb = get_musicbert_last_hidden_embedding(model, vocab, file) # output h is already the h[:, 0] (CLS token)
        avged_embs.append(avged_emb)
    
    avged_embs = torch.cat(avged_embs, dim=0)
    ref_emb = nn.functional.normalize(ref_emb, dim=1)
    avged_embs = nn.functional.normalize(avged_embs, dim=1)
    logits = ref_emb @ avged_embs.T # cosine similarity calculation: just matrix multiplication
    ret_item = pd.Series(logits.squeeze(0).detach().cpu().numpy(), index=dup_files)
    ret_item = ret_item.sort_values(ascending=False)
    
    return ret_item
       
def get_tokenized_seq(midi_path, file_ext, vocab):
    npy_path = midi_path.with_name(midi_path.stem + file_ext)
    if os.path.exists(npy_path):
        seq = np.load(npy_path)
    else:
        midi = Score(midi_path)
        tokens = vocab(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
        seq = np.array(tokens)
        
        np.save(npy_path, seq)
    return seq
    
def get_subseqs(seq, min_seq_len=100, max_seq_len=1024, midi_path=None, file_ext=None, out_dir=None):
    subseqs = split_seq_in_subsequences(seq, min_seq_len=min_seq_len, max_seq_len=max_seq_len)
    for i, subseq in enumerate(subseqs):
        if midi_path is not None and file_ext is not None:
            midi_path = Path(midi_path)
            fp = midi_path.with_name(f'{midi_path.stem}_subseq_{i}.mid')
            if out_dir is not None:
                artist_dir = midi_path.parent.name
                out_path = Path(out_dir) / artist_dir / fp.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fp = out_path
                # print(fp)
                recon_midi = vocab(subseq)
                recon_midi.dump_midi(fp)
    
    return subseqs

def add_bos_eos(x, vocab):
    l = len(vocab._vocab_base)
    return torch.cat([
        torch.full((1, l), vocab['BOS_None']),
        x,
        torch.full((1, l), vocab['EOS_None'])
    ], dim=0)


# Loads a midi, converts to tokens, and back to a MIDI
def get_musicbert_last_hidden_embedding(model, vocab, midi_path, file_ext='_octuple_1d.npy', embedding_mode='CLS'):
    seq = get_tokenized_seq(midi_path, file_ext, vocab)
    assert len(seq) != 0, f"Empty sequence for {midi_path}"
    # 곡의 맨 처음 1024 token에 대해서 hidden state 저장
    min_seq_len = 100
    subseqs = get_subseqs(seq, min_seq_len=min_seq_len, max_seq_len=1022, midi_path=midi_path, file_ext=file_ext)
    
    if len(seq) < min_seq_len:
        print(f"Sequence length is too short: {len(seq)}: {midi_path}")
        with open('lmd_full_inference_error_log.txt', 'a+') as f:
            f.write(f"Sequence length is too short: {len(seq)}: {midi_path}\n")
        return None
    
    if len(subseqs) == 0:
        print(f"Subsequence length is too short: {len(seq)}: {midi_path}")
        with open('lmd_full_inference_error_log.txt', 'a+') as f:
            f.write(f"Subsequence length is too short: {len(seq)}: {midi_path}\n")
        return None
    
    with torch.no_grad():
        for i, subseq in enumerate(subseqs):
            if i > 0:
                break
            input_data = torch.tensor(subseq)
            input_data = add_bos_eos(input_data, vocab)
            input_data = input_data.unsqueeze(0).to(model.device)
            if embedding_mode=='CLS':
                logits, h, h_list = model(input_data)
            elif embedding_mode=='mean':
                h = model.get_last_all_hidden(input_data)
                h = h.mean(dim=1)
            else:
                raise ValueError(f"Invalid embedding_mode: {embedding_mode}")
    return h
