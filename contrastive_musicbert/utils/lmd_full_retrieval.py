import os
import tarfile
import torch
import numpy as np
from pathlib import Path

def get_embeddings_and_refs(embedding_dir, embedding_file, embedding_refs, sort, mode='list'):
    set_embs = torch.load(embedding_dir + embedding_file)

    # load the refs
    with open(embedding_dir + embedding_refs, 'r') as f:
        refs = f.readlines()
        
    refs = [change_ref_path(ref, mode) for ref in refs]
    print(set_embs.shape)
    print(refs[:5])

    return set_embs, refs

def compress_directory(folder_path, archive_name):
    with tarfile.open(archive_name, "w:gz") as tar:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=folder_path) 
                tar.add(full_path, arcname=arcname)

def dfs(graph, node, visited, cluster_id, node_to_cluster):
    if node not in visited:
        visited.add(node)
        node_to_cluster[node] = cluster_id  # Assign the current cluster ID to the node
        for neighbour in graph[node]:
            dfs(graph, neighbour, visited, cluster_id, node_to_cluster)

def find_clusters(graph):
    visited = set()
    node_to_cluster = {}  # This dictionary will map each node to its cluster ID
    cluster_id = 0  # Initialize cluster ID

    for node in graph:
        if node not in visited:
            dfs(graph, node, visited, cluster_id, node_to_cluster)
            cluster_id += 1  # Increment the cluster ID for the next cluster

    return node_to_cluster