import os
import copy
import time
import torch
import random
import pickle
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
import json

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from miditok.pytorch_data import split_seq_in_subsequences
from .augmentator import Augmentator

class DataModule(pl.LightningDataModule):
    """
    Refer to https://github.com/dvruette/figaro/blob/2b253eb0476453e197ee0599a5c58f87d82a3890/src/datasets.py
    """

    def __init__(
        self,
        file_list,
        vocab,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        masking=0.8,
        replace=0.1,
        phase="train",
        masking_strategy=['element', 'compound', 'bar'],
        cache_path="dataset_cache/dataset_cache.pkl",
        debug=False,
    ):
        super().__init__()

        self.file_list = file_list
        self.split = phase
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.masking = masking
        self.replace = replace

        self.shuffle = True if self.split == 'train' else False

        self.vocab = vocab
        self.masking_strategy = masking_strategy
        self.cache_path = Path(cache_path)
        self.debug = debug
        self.setup()

    def setup(self):
        self.file_list = [Path(file) for file in self.file_list]
        
        self.data = DatasetSampler(
            self.file_list,
            min_seq_len=100,
            max_seq_len=1022,
            bos_token=self.vocab["BOS_None"], 
            eos_token=self.vocab["EOS_None"],
            n_types=len(self.vocab._vocab_base),
            sample_key_name="input_ids",
            one_token_stream=self.vocab.one_token_stream,
            split=self.split,
            cache_path=self.cache_path,
            debug=self.debug,
        )

        self.collator = SeqCollator(
            self.vocab,
            self.vocab["PAD_None"], 
            self.vocab["BOS_None"], 
            self.vocab["EOS_None"],
            self.masking,
            self.replace,
            "input_ids",
            "input_ids",
            self.masking_strategy,
        )

    def return_dataloader(self):
        return DataLoader(
            self.data,
            collate_fn=self.collator,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


class DatasetSampler(Dataset):
    def __init__(
        self, 
        file_list: list, # Path
        min_seq_len: int,
        max_seq_len: int,
        bos_token: int,
        eos_token: int,
        n_types: int,
        one_token_stream: bool = False,
        sample_key_name: str = "input_ids",
        cache_path: Path = Path("dataset_cache/dataset_cache.pkl"),
        split: str = None,
        debug: bool = False,
        ) -> None:
        
        self.file_list = file_list
        self.file_list = [Path(self.file_list) for self.file_list in self.file_list] # for pathlib issue

        self.sample_indices = []
        self.file_indices = [-1 for _ in range(len(file_list))]
        self.one_token_stream = one_token_stream
        self.sample_key_name = sample_key_name
        
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.n_types = n_types
        
        if split is not None:
            cache_path = cache_path.parent / f"{cache_path.stem}_{split}_{debug}.pkl"
        
        # if cache exists, load it
        if cache_path.exists():
            print(f'Loading cache from {cache_path}...')
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
            self.sample_indices = cache['sample_indices']
            self.file_indices = cache['file_indices']

        else:
            # else, create cache
            print(f'Creating cache at {cache_path}...')
            short_seqs = 0
            for f_id, file_path in tqdm(
                enumerate(file_list),
                desc=f"Checking data: {file_list[0].parent}",
                miniters=int(len(file_list) / 20),
                maxinterval=480,
                total=len(file_list),
            ):
                self.file_indices[f_id] = len(self.sample_indices)
                with file_path.open() as json_file:
                    tokens = json.load(json_file)
                tokens_ids = tokens["ids"]

                # Cut tokens in samples of appropriate length
                if self.one_token_stream:
                    tokens_ids = [tokens_ids]
                
                for seq in tokens_ids:
                    subseqs = split_seq_in_subsequences(seq, min_seq_len, max_seq_len)
                    if len(subseqs) == 1:
                        short_seqs += 1
                        
                    subseq_idx = 0
                    for subseq in subseqs:
                        self.sample_indices.append((f_id, subseq_idx, subseq_idx + len(subseq)))
                        subseq_idx += len(subseq)
                    
            print('Short sequences:', short_seqs)
            cache = {
                'sample_indices': self.sample_indices,
                'file_indices': self.file_indices,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
                print('Cache saved')

    def __len__(self):
        return len(self.sample_indices)
    

    def get_file_seg_indices(self, file_index):
        if file_index == len(self.file_indices) - 1:
            # last segment
            assert self.file_indices[file_index] < len(self.sample_indices)
            return self.file_indices[file_index], len(self.sample_indices)
        else:
            return self.file_indices[file_index], self.file_indices[file_index + 1]

    def add_bos_eos(self, x):
        return torch.cat([
            torch.full((1, self.n_types), self.bos_token),
            x,
            torch.full((1, self.n_types), self.eos_token)
        ], dim=0)


    def __getitem__(self, idx):
        f_id, start_idx, end_idx = self.sample_indices[idx]
        
        file_path = self.file_list[f_id]
        file_index = self.file_indices[f_id]

        file_seg_start, file_seg_end = self.get_file_seg_indices(f_id)
        # choose random sample from the same file
        # if only one segment exists, choose itself
        if file_seg_end - file_seg_start == 1:
            neighbor_f_id, neighbor_start_idx, neighbor_end_idx = f_id, start_idx, end_idx
        else:
            while True:
                rand_idx = random.randint(file_seg_start, file_seg_end - 1) # inclusive
                if rand_idx != idx:
                    break
            neighbor_f_id, neighbor_start_idx, neighbor_end_idx = self.sample_indices[rand_idx]
        
        with open(file_path, "r") as f:
            tokens = json.load(f)
        
        tokens_ids = tokens["ids"]
        
        assert len(tokens_ids) >= end_idx, f"tokens_ids length: {len(tokens_ids)}, end_idx: {end_idx}"
        x = torch.LongTensor(tokens_ids[start_idx:end_idx])
        neighbor_x = torch.LongTensor(tokens_ids[neighbor_start_idx:neighbor_end_idx])
        x = self.add_bos_eos(x)
        neighbor_x = self.add_bos_eos(neighbor_x)

        return {
            self.sample_key_name: x,
            "neighbor": neighbor_x,
            "f_id": f_id # since we can get the file path using self.file_list[f_id], f_id is enough
        }


class SeqCollator:
    """
    Refer to https://github.com/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.py
    """

    def __init__(self, vocab, pad_token, bos_token, eos_token, masking, replace, 
                 inputs_kwarg_name: str = "input_ids", 
                 labels_kwarg_name: str = "labels",
                 masking_strategy=['element', 'compound', 'bar']):
        self.vocab = vocab
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.augmentator = Augmentator(vocab)

        self.masking = masking
        self.replace = replace
        self.random_token_prob = 0.1
        self.leave_unmasked_prob = 0.1
        self.mask_prob = 0.15
        self.return_masked_tokens = True
        self.pad_idx = self.vocab.special_tokens.index('PAD_None') 
        self.mask_idx = self.vocab.special_tokens.index('MASK_None') 
        
        self.inputs_kwarg_name = inputs_kwarg_name
        self.labels_kwarg_name = labels_kwarg_name
        self.mask_whole_words = False
        self.masking_strategy = masking_strategy

    def get_masked_seq(self, feature, mask_strategy=['element', 'compound', 'bar']):
        inputs = copy.deepcopy(feature[self.inputs_kwarg_name])       
        n_tokens = [len(self.vocab._vocab_base[i]) for i in range(len(self.vocab._vocab_base))]
        n_types = len(self.vocab._vocab_base)
        for i in range(len(n_tokens)):
            assert max(inputs[:, i]) < n_tokens[i]
        
        # make the sequence as 1-dimensional array
        item = inputs.flatten()
        sz = len(item)
        assert (
            self.mask_idx not in item
        ), "Dataset contains mask_idx (={}), this is not expected!".format(
            self.mask_idx,
        )

        def generate_mask(sz, prob):
            mask_n = np.random.rand(sz)
            mask_s = np.zeros(sz, dtype=np.int8)
            mask_s += mask_n < prob * \
                (self.random_token_prob)  # 3 -> random
            mask_s += mask_n < prob * \
                (self.random_token_prob +
                    self.leave_unmasked_prob)  # 2 -> original
            mask_s += mask_n < prob * 1.00  # 1 -> [mask]
            return mask_s
        mask_prob = self.mask_prob
        mask = np.zeros_like(item, dtype=np.int8)
        # mask bos eos tokens (compound)
        mask[:n_types] = np.repeat(generate_mask(1, mask_prob), n_types)
        # mask bos eos tokens (compound)
        mask[-n_types:] = np.repeat(generate_mask(1, mask_prob), n_types)
        strategy = np.random.choice(mask_strategy) # why mask is randomly chosen?
        if strategy == 'element':  # element level mask
            mask[n_types:-n_types] = np.repeat(generate_mask(sz - 2 * n_types, mask_prob), 1) # consider bos and eos tokens
        elif strategy == 'compound':
            mask[n_types:-n_types] = np.repeat(generate_mask(sz // n_types - 2, mask_prob), n_types) # consider bos and eos tokens
        elif strategy == 'bar':
            max_bars = 256 # octuple original implementation
            max_instruments = 127 # octuple original implementation
            bar_idx = self.vocab.vocab_types_idx['Bar']
            ins_idx = self.vocab.vocab_types_idx['Program']
            offset = len(self.vocab.special_tokens)
            mask[n_types:-n_types] = generate_mask((max_bars * max_instruments + len(self.vocab)) * n_types, mask_prob).reshape(-1, n_types)[((item[n_types+bar_idx:-n_types+bar_idx: n_types] - offset) * max_instruments) + (item[n_types+ins_idx:-n_types+ins_idx:n_types] - offset)].flatten()
        else:
            raise NotImplementedError(f"Masking strategy {strategy} is not implemented")

        masked_item = []
        for i in range(len(n_tokens)):
            masked_it = np.random.choice(n_tokens[i], sz//len(self.vocab._vocab_base))
            masked_item.append(masked_it)
        masked_item = np.stack(masked_item, axis=1)
        
        for i in range(len(n_tokens)):
            assert max(masked_item[:, i]) < n_tokens[i]
            
        masked_item = masked_item.flatten()
        assert masked_item.shape == item.shape
        
        item_view = item.view(-1, len(self.vocab._vocab_base))
        
        set_original = np.isin(mask, [0, 2])
        masked_item[set_original] = item[set_original]
        set_mask = np.isin(mask, [1])
        masked_item[set_mask] = self.mask_idx
        masked_inputs = torch.from_numpy(masked_item.reshape(-1, len(self.vocab._vocab_base)))
        
        for i in range(len(n_tokens)):
            assert max(masked_inputs[:, i]) < n_tokens[i]
        
        new_item = item.numpy()[:]
        new_item[mask == 0] = self.pad_idx
        masked_outputs = torch.from_numpy(new_item.reshape(-1, len(self.vocab._vocab_base)))
        
        return masked_inputs, masked_outputs

    def get_augment_seq(self, feature):
        events = copy.deepcopy(feature[self.inputs_kwarg_name])

        events, _ = self.augmentator(events)
        return events

    def adjust_bar_shift(self, feature):
        events = feature[self.inputs_kwarg_name]
        events = self.augmentator.adjust_bar_shift(events)                
        feature[self.inputs_kwarg_name] = events

        return feature

    def get_neighbor_seq(self, feature):
        events = feature['neighbor']
        events = self.augmentator.adjust_bar_shift(events)
        events, _ = self.augmentator(events)
        return events

    def __call__(self, features):
        (
            x_list,
            x_aug_list,
            x_neigh_list,
            x_mask_list,
            y_mask_list,
            file_list,
        ) = ([] for _ in range(6))

        (
            inst_list,
            chord_list,
            tempo_list,
            vel_list,
            dur_list,
            gp_list,
        ) = ([] for _ in range(6))

        batch = {}
        for feature in features:
            ### original inputs
            feature = self.adjust_bar_shift(feature) # original musicbert rather made random bar shift
            x = copy.deepcopy(feature[self.inputs_kwarg_name])

            ### augment inputs
            x_aug = self.get_augment_seq(feature)
            
            ### neighbor inputs
            x_neigh = self.get_neighbor_seq(feature)
            
            ### masking inputs
            x_mask, y_mask = self.get_masked_seq(feature, self.masking_strategy)
            
            # convert events to index
            assert x.dtype == torch.long

            x_list.append(x)
            x_aug_list.append(x_aug)
            x_neigh_list.append(x_neigh)
            x_mask_list.append(x_mask)
            y_mask_list.append(y_mask)

            file_list.append(feature["f_id"])

        x_list = pad_sequence(x_list, batch_first=True, padding_value=self.pad_token)
        x_aug_list = pad_sequence(x_aug_list, batch_first=True, padding_value=self.pad_token)
        x_neigh_list = pad_sequence(x_neigh_list, batch_first=True, padding_value=self.pad_token)
        x_mask_list = pad_sequence(x_mask_list, batch_first=True, padding_value=self.pad_token)
        y_mask_list = pad_sequence(y_mask_list, batch_first=True, padding_value=self.pad_token)

        batch["x"] = x_list
        batch["x_aug"] = x_aug_list
        batch["x_neigh"] = x_neigh_list
        batch["x_mask"] = x_mask_list
        batch["y_mask"] = y_mask_list
        batch["file_name"] = file_list

        return batch