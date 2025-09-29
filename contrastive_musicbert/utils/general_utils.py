import os
import glob
import torch
import random
import numpy as np
import tarfile

from collections import Counter
from joblib import parallel_backend

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error


def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # it may slow computing performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def folder_to_file(folder_list):
    file_list = [glob.glob(os.path.join(folder, "*.json")) for folder in folder_list]
    file_list = [elem for sublist in file_list for elem in sublist]
    if len(file_list) == 0:
        file_list = folder_list

    return file_list


def folder_to_multiple_file(folder_list, k=1):
    results = []
    origin_k = k

    for folder in folder_list:
        file_list = glob.glob(os.path.join(folder, "*"))

        k = origin_k
        if k > len(file_list):
            k = len(file_list)

        file = random.sample(file_list, k)
        results += file

    return results


def dataset_split(dataset, train_ratio=0.8, val_ratio=0.1):
    num_train = int(len(dataset) * train_ratio)
    num_val = int(len(dataset) * val_ratio)

    train_idx = num_train
    val_idx = num_train + num_val

    train_set = dataset[:train_idx]
    val_set = dataset[train_idx:val_idx]
    test_set = dataset[val_idx:]

    return train_set, val_set, test_set


def to_numpy(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.detach().cpu().numpy()
        else:
            x = x.numpy()
    elif isinstance(x, list):
        x = np.asarray(x)

    return x

def compress_directory(folder_path, archive_name):
    with tarfile.open(archive_name, "w:gz") as tar:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=folder_path)  # 원본 디렉토리 구조를 유지
                tar.add(full_path, arcname=arcname)