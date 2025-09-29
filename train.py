from pathlib import Path
import glob
import json
import random
import time
import pickle
import numpy as np
from datetime import datetime

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from contrastive_musicbert.utils.general_utils import set_manual_seed, folder_to_file, folder_to_multiple_file, dataset_split
    
@hydra.main(version_base=None, config_path="./yamls/", config_name="train")
def main(config: DictConfig):
    from contrastive_musicbert.data.data import DataModule
    from contrastive_musicbert.model.BERT import BERT_Lightning
    from contrastive_musicbert.data.custom_octuple import CustomOctuple
    vocab = CustomOctuple()
    
    random_seed = config.random_seed
    set_manual_seed(random_seed)

    torch.set_float32_matmul_precision("high")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('use ', device)

    folder_path = Path(config.folder_path)
    split_path = Path(config.split_path)

    print(f"load dataset from {folder_path}")

    if not split_path.exists():
        folder_list_paths = list(folder_path.glob("*/*"))
        folder_list = list(map(str, folder_list_paths))
        print(f'folder_list : {len(folder_list)}')
        
        train_folder, val_folder, test_folder = dataset_split(folder_list, train_ratio=0.98, val_ratio=0.01)
        print(f'train_folder : {len(train_folder)}, val_folder : {len(val_folder)}, test_folder : {len(test_folder)}')
        train_files = folder_to_file(train_folder)
        val_files = folder_to_file(val_folder)
        test_files = folder_to_file(test_folder)

        with open(config.split_path, 'wb') as f:
            pickle.dump([train_files, val_files, test_files], f)
        print(f'{config.split_path} is saved')
    else:
        print(f'load {config.split_path}')
        with open(config.split_path, 'rb') as f:
            train_files, val_files, test_files = pickle.load(f)  

    random.shuffle(train_files)
    
    if config.debug:
        train_files = train_files[:10]
        val_files = val_files[:10]
        test_files = test_files[:10]
    print(
        f"train_files : {len(train_files)}, val_files : {len(val_files)}, test_files : {len(test_files)}"
    )

    print(
        f'pin_memory : {config.pin_memory}'
    )

    train_module = DataModule(
        train_files,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        masking=config.masking,
        replace=config.replace,
        phase="train",
        pin_memory=config.pin_memory,
        vocab=vocab,
        masking_strategy=config.masking_strategy,
        cache_path=config.cache_path,
        debug=config.debug,
    )

    val_module = DataModule(
        val_files,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        phase="val",
        pin_memory=config.pin_memory,
        vocab=vocab,
        masking_strategy=config.masking_strategy,
        cache_path=config.cache_path,
        debug=config.debug,
    )

    test_module = DataModule(
        test_files,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        phase="test",
        pin_memory=config.pin_memory,
        vocab=vocab,
        masking_strategy=config.masking_strategy,
        cache_path=config.cache_path,
        debug=config.debug,
    )

    load_start_time = time.time()
    train_set = train_module.return_dataloader()
    val_set = val_module.return_dataloader()
    test_set = test_module.return_dataloader()
    print(f"DataLoader loading time : {time.time() - load_start_time} sec")

    config_dict = OmegaConf.to_container(config, resolve=True)
    
    output_dir = HydraConfig.get().run.dir # hydra output directory
    model_name = "-".join(config.mode)
    model_name = "BERT-" + model_name + "-{epoch}-{val_loss:.4f}"
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint = ModelCheckpoint(
        filename=model_name,
        monitor="val_loss",
        mode="min",
        dirpath=str(output_dir / "checkpoints"),
        save_last=True,
        save_top_k=2,
    )

    logger = WandbLogger(
        project=f"MusicBERT",
        save_dir = output_dir,
    )

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
        total_steps=config.total_steps,
    ).to(device)
    
    trainer = pl.Trainer(
        num_nodes=1,
        precision=16,
        max_epochs=config.epochs,
        accelerator="gpu",
        devices=config.gpus,
        callbacks=[lr_monitor, checkpoint],
        logger = logger,
        val_check_interval=config.val_check_interval
    )
    
    trainer.fit(model, train_set, val_set)

if __name__ == "__main__":
    main()
