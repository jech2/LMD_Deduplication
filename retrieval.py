import torch
from pathlib import Path
import numpy as np
from torch import nn
import pandas as pd
import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from contrastive_musicbert.utils.midi_synth_dir import save_wav_from_performance_midi
from contrastive_musicbert.utils.general_utils import compress_directory
from contrastive_musicbert.utils.metadata_handler import MetadataHandler
from contrastive_musicbert.metadata.filepaths import LMD_FULL_FILE_LIST, LMD_CLEAN_FILE_LIST, LMD_CLEAN_DUP_FILE_LIST

# get the torch embedding and average it (1 x n_seq x d_hidden) -> (1 x d_hidden)
def get_embedding(file_path, avg_embedding=True, device='cuda'):
    file_path = Path(file_path)
    if file_path.suffix == '.npy':
        last_hidden = np.load(file_path, allow_pickle=True)
        last_hidden = torch.from_numpy(last_hidden)
    elif file_path.suffix == '.pt':
        last_hidden = torch.load(file_path, map_location=torch.device(device))
    else:
        raise ValueError(f'Unknown file extension: {file_path.suffix}')
    
    if avg_embedding:
        if len(last_hidden.shape) == 2:
            return last_hidden
        last_hidden = torch.mean(last_hidden, dim=1)
    
    return last_hidden

# get all dataset embedding (n_seg x d_hidden)
def get_all_embedding(refs, ref_dir, avg_embedding=True, file_ext='.pkl'):
    embs = []
    valid_key_lists = []
    error_cnts = 0
    m = MetadataHandler()
    for i, key in tqdm(enumerate(refs)):
        emb_path = m.get_artist_and_title_str_to_filepath(key, ref_dir, file_ext)
        try:
            embedding = get_embedding(emb_path, avg_embedding)
            embs.append(embedding)
            valid_key_lists.append(key)
        except Exception as e:
            error_cnts += 1
            continue

    print(f'error count: {error_cnts}. Total valid refs: {len(valid_key_lists)}')
    embs = torch.cat(embs, dim=0)
    print(f'total embedding shape: {embs.shape}')
    return embs, valid_key_lists

def retrieval(query, set_embs, refs, avg_embedding):
    qry_emb_file, qry_midi_file = query
    with torch.no_grad():
        qry_embs = get_embedding(qry_emb_file, avg_embedding)
        qry_embs = nn.functional.normalize(qry_embs, dim=1)
        set_embs = nn.functional.normalize(set_embs, dim=1)
        
        logits = qry_embs @ set_embs.T # cosine similarity calculation: just matrix multiplication
        ret_item = pd.Series(logits.squeeze(0).detach().cpu().numpy(), index=refs)
        torch.cuda.empty_cache()
    instance = {}
    metadata = {}
    similarities = {}
   
    for idx, _id in enumerate(ret_item.sort_values(ascending=False).head(10).index):
        meta_info = f'{_id} | {ret_item[_id]:.3f}'
        similarities[_id] = float(ret_item[_id])
        
    return instance, metadata, similarities, ret_item

def get_query_lists(ref_queries, ref_dir, file_ext, m):
    query = []
    for ref_query in ref_queries:
        qry_emb_file = Path(m.get_artist_and_title_str_to_filepath(ref_query, ref_dir, file_ext)) # embedding file extension is ./dataset/clean_midi/artist/song + ext
        try:
            qry_midi_file = Path(m.get_artist_and_title_str_to_filepath(ref_query, ref_dir, '.mid')) # assuming the file is always ./dataset/clean_midi/artist/song.mid
        except:
            qry_midi_file = None
        query += [(qry_emb_file, qry_midi_file)]
    return query

def get_file_lists(m, dataset_name):
    if dataset_name == 'lmd_dup':
        return m.get_lmd_clean_dup_file_list()
    elif dataset_name == 'lmd_clean':
        return m.get_lmd_clean_file_list()
    elif dataset_name == 'lmd_full':
        return m.get_lmd_full_file_list()
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')

def get_duplicate_similarities(ret_item, target_file_str, m):
    duplicate_meta = m.get_all_version_count_meta()
    assert target_file_str in duplicate_meta.keys()
    duplicate_meta = duplicate_meta[target_file_str]
    
    similarities = {}

    # get the similarity of the duplicates
    for duplicate in duplicate_meta['versions']:
        if duplicate not in ret_item.keys():
            continue
        similarities[duplicate] = float(ret_item[duplicate])
        
    return similarities

def save_dictionary_and_plot(dict_obj, dict_fp, plot_fp, plot_title):
    with open(dict_fp, "w") as json_file:
        json.dump(dict_obj, json_file, indent=4)
        
    data = [list(dict_obj[key].values()) for key in dict_obj]
    v_data = list(itertools.chain(*data))
    plt.figure(figsize=(10, 6))
    plt.violinplot(v_data)
    plt.title(plot_title)
    plt.xlabel('Group')
    plt.ylabel('Value')
    plt.ylim((0, 1))
    plt.grid(True)
    plt.savefig(plot_fp, dpi=200)


@hydra.main(version_base=None, config_path="./yamls/", config_name="retrieval")
def main(config: DictConfig):
    m = MetadataHandler()
    queries = get_file_lists(m, config.qry_dataset)
    refs = get_file_lists(m, config.ref_dataset)

    query = get_query_lists(queries, m.get_ref_dir(config.qry_dataset), config.file_ext, m) # query = [(emb_file, midi_file), ...]

    print('get all embedding files...')

    new_target_base_dir = Path(f'{config.result_dir}/{config.qry_dataset}_to_{config.ref_dataset}_{config.thres}_{config.file_ext}/')
    if new_target_base_dir.exists():
        shutil.rmtree(new_target_base_dir)
    new_target_base_dir.mkdir(parents=True, exist_ok=True)
    
    set_embs_fp = new_target_base_dir / f'embeddings.pt'
    valid_refs_fp = new_target_base_dir / f'refs.txt'

    if set_embs_fp.exists() and valid_refs_fp.exists():
        print('Load embeddings and ref list..')
        set_embs = torch.load(set_embs_fp)
        with open(valid_refs_fp, 'r') as f:
            valid_refs = [line.strip() for line in f]
    else:
        with torch.no_grad():
            set_embs, valid_refs = get_all_embedding(refs, m.get_ref_dir(config.ref_dataset), file_ext=config.file_ext, avg_embedding=config.avg_embedding)
        torch.save(set_embs, set_embs_fp)

        with open(valid_refs_fp, 'w') as f:
            for ref in valid_refs:
                f.write(f'{ref}\n')

        print('Embeddings and ref list are saved.')
                
    if config.qry_dataset == 'lmd_full':
        print('Since the lmd full is too large, we will not run the retrieval for lmd_full.')
        exit()

    if config.save_embedding_list_only:
        print('Save embeddings list only.')
        exit()
  
    print('start retrieval...')

    ### Retrieval
    all_top_10_similarities = {}
    duplicates_similarities = {}
    for q in tqdm(query):
        try:
            instance, metadata, top_10_similarities, ret_item = retrieval(q, set_embs, valid_refs, avg_embedding=config.avg_embedding)
            _, qry_midi_file = q
            qry_file_str = m.get_filepath_to_artist_and_title_str(qry_midi_file)
            all_top_10_similarities[qry_file_str] = top_10_similarities
            if config.qry_dataset == 'lmd_dup' and config.ref_dataset == 'lmd_clean':
                duplicates_similarities[qry_file_str] = get_duplicate_similarities(ret_item, qry_file_str, m)
        
            if config.save_samples:
                ### Save the samples that have similarity higher than the threshold
                target_fp = q[1]
                for ref_song, sim in top_10_similarities.items():
                    ref_fp = m.get_artist_and_title_str_to_filepath(ref_song, m.get_ref_dir(config.ref_dataset), '.mid')
                    if sim >= config.thres:
                        ref_fp = Path(ref_fp)
                        target_fp = Path(target_fp)
                        
                        new_target_dir = new_target_base_dir / target_fp.parent.name / target_fp.name
                        new_target_dir.mkdir(parents=True, exist_ok=True)
                        
                        new_ref_dir = new_target_dir / ref_fp.parent.name
                        new_ref_dir.mkdir(parents=True, exist_ok=True)
                        new_ref_fp = new_ref_dir / (ref_fp.stem + f'_{sim:.3f}' + ref_fp.suffix)
                        
                        ref_mid_fp = ref_fp.parent / (ref_fp.stem.split('_')[0] + '.mid')
                        new_ref_mid_fp = new_ref_fp.parent / (new_ref_fp.name.replace(ref_fp.suffix, '.mid'))
                        shutil.copyfile(src=ref_mid_fp, dst=new_ref_mid_fp)
        except Exception as e:
            print(f'retrieval failed for {q}: {e}')
            with open(new_target_base_dir / 'error_log.txt', 'a+') as f:
                f.write(f'retrieval failed for {q}: {e}\n')
            continue
        
    # dictionary save
    dict_fp = new_target_base_dir / 'top_10_similarities.json'
    png_fp = new_target_base_dir / 'top_10_similarities.png'
    save_dictionary_and_plot(all_top_10_similarities, dict_fp, png_fp, f'Violin Plot of similarity of {config.qry_dataset} vs {config.ref_dataset}')
    if config.qry_dataset == 'lmd_dup' and config.ref_dataset == 'lmd_clean':
        dict_fp = new_target_base_dir / 'duplicates_similarities.json'
        png_fp = new_target_base_dir / 'duplicates_similarities.png'
        save_dictionary_and_plot(duplicates_similarities, dict_fp, png_fp, f'Violin Plot of similarity of duplicates of {config.qry_dataset} vs {config.ref_dataset}')

    # compress the directory
    folder = str(new_target_base_dir)
    if config.compress_dir:
        compress_directory(folder, folder + '.tar.gz')

if __name__ == '__main__':
    main()