from pathlib import Path
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


def get_similarities_from_embeddings(ref_dir, embedding_file):
    with open(ref_dir / "refs.txt", "r") as f:
        refs = f.readlines()

    refs = [k.strip() for k in refs]
    set_embs = torch.load(ref_dir / embedding_file)

    set_embs = nn.functional.normalize(set_embs, dim=1)

    similarity = torch.mm(set_embs, set_embs.t())

    df = pd.DataFrame(similarity.cpu().numpy(), index=refs, columns=refs)

    return df

def get_similarity_from_beat_entropy(ref_dir, sim_file):
    df = pd.read_pickle(ref_dir / sim_file)
    # all items are 1-distance as similarity
    df = 1 - df
    df.fillna(1, inplace=True)
    return df

def reset_relative_path_index_col(df, mode="ours"):
    if 'clean_midi' in df.index[0]:
        # artist__title
        df.index = [
            Path(index).parent.name + '__' + Path(index).stem
            for index in df.index
        ]
    
    if 'clean_midi' in df.columns[0]:
        df.columns = [
            Path(column).parent.name + '__' + Path(column).stem
            for column in df.columns
        ]
        
    print(df.index[0])
    print(df.columns[0])
    
    assert '__' in df.index[0], f"Index is not in the format of artist__title: {df.index[0]}"
    assert '__' in df.columns[0], f"Columns is not in the format of artist__title: {df.columns[0]}"

    return df


# get the intersection of queries and refs
def get_intersections_queries_refs_all(df_lists):
    queries_lists = []
    refs_lists = []
    for df in df_lists:
        queries_lists.append(list(df.index))
        refs_lists.append(list(df.columns))
        print(f"Queries: {len(queries_lists[-1])}", queries_lists[-1][0])
        print(f"Refs: {len(refs_lists[-1])}", refs_lists[-1][0])

    # get the intersection of queries_lists
    queries_intersection = set(queries_lists[0])
    for queries in queries_lists:
        queries_intersection = queries_intersection.intersection(queries)

    if len(queries_intersection) == 0:
        print('something is wrong')


    # get the intersection of refs_lists
    refs_intersection = set(refs_lists[0])
    for refs in refs_lists:
        refs_intersection = refs_intersection.intersection(refs)

    # queries should be in the refs
    queries_intersection = queries_intersection.intersection(refs_intersection)

    print(f"Queries intersection: {len(queries_intersection)}")
    print(f"Refs intersection: {len(refs_intersection)}")

    return queries_intersection, refs_intersection

def filter_queries_refs_all(
    dfs, all_model_names, queries_intersection, refs_intersection, out_dir
):
    print(
        f"filtering queries and refs... as {len(queries_intersection)} queries and {len(refs_intersection)} refs"
    )

    for model_name, df in zip(all_model_names, dfs):
        # save filtered npy
        output_file = out_dir / model_name / "filtered_similarities_with_queries.pkl"
        if output_file.exists():
            print(f"Already exists: {output_file}")
            continue
        else:
            df_filtered = filter_queries_refs(df, queries_intersection, refs_intersection)
            df_filtered = df_filtered.clip(0, 1)
            print(f"Saving to {output_file}")
            with open(output_file, "wb") as f:
                pickle.dump(df_filtered, f)


# filter out the queries and refs that are not in the intersection
def filter_queries_refs(df, queries_intersection, refs_intersection):
    # Convert the sets to lists before using them as indexers
    queries_list = list(queries_intersection)
    refs_list = list(refs_intersection)

    return df.loc[queries_list, refs_list]

@hydra.main(version_base=None, config_path="./yamls/", config_name="evaluation")
def main(config: DictConfig):

    with open(config.eval_models_path, "r") as f:
        model_dict = json.load(f)

    all_model_dirs = list(model_dict.values())


    file_fps = ['embeddings.pt', 'beat_similarity_df_artist.pkl', 'all_similarities_with_query.pkl']

    out_base_dir = Path(config.eval_dir) 
    dfs = []

    for file_dir in all_model_dirs:
        available_fp = None
        file_dir = Path(config.inp_dir) / file_dir
            
        # save the df
        out_dir = out_base_dir / file_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for file_fp in file_fps:
            target_fp = file_dir / file_fp
            if target_fp.exists():
                available_fp = file_fp
                break
        assert available_fp is not None, f"File not found in {file_dir}"
        
        if available_fp == 'embeddings.pt':
            df = get_similarities_from_embeddings(file_dir, available_fp)
        elif available_fp == 'beat_similarity_df_artist.pkl':
            df = get_similarity_from_beat_entropy(file_dir, available_fp)
        else:    
            with open(file_dir / available_fp, 'rb') as f:
                df = pickle.load(f)
                
        df = reset_relative_path_index_col(df)
        print(df.shape, file_dir)
        
        with open(out_dir / 'all_similarities_with_query.pkl', 'wb') as f:
            pickle.dump(df, f)        
        
        with open(out_dir / 'queries.txt', 'w') as f:
            f.write('\n'.join(df.index))
            
        with open(out_dir / 'refs.txt', 'w') as f:
            f.write('\n'.join(df.columns))
        
        dfs.append(df)


    filt_q_r_fp = out_base_dir / "filtered_queries_refs.pkl"
    if filt_q_r_fp.exists():
        with open(filt_q_r_fp, "rb") as f:
            queries_intersection, refs_intersection = pickle.load(f)
    else:
        queries_intersection, refs_intersection = get_intersections_queries_refs_all(dfs)
        # exit()
        assert len(queries_intersection) > 0, "No intersection found in queries"
        assert len(refs_intersection) > 0, "No intersection found in refs"
        with open(filt_q_k_fp, "wb") as f:
            pickle.dump((queries_intersection, refs_intersection), f)
        
    print('intersections', len(queries_intersection), len(refs_intersection))
    all_model_dirs_names = [Path(file_dir).name for file_dir in all_model_dirs]
    filter_queries_refs_all(
        dfs, all_model_dirs_names, queries_intersection, refs_intersection, out_base_dir
    )

if __name__ == "__main__":
    main()