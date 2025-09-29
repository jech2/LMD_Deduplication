# On the De-duplication of the Lakh MIDI Dataset

This is the official repository for our paper: **“On the De-duplication of the Lakh MIDI Dataset”** (ISMIR 2025)

## Links
- **[ISMIR miniconf](https://ismir2025program.ismir.net/poster_188.html)** [[arXiv](https://arxiv.org/abs/2509.16662)]
- **Interactive Demo:** [Demo Page](https://jech2.github.io/LMD_Deduplication/)
- **Original Dataset:** [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)
- **Pre-trained CAugBERT Checkpoint:** [HuggingFace](https://huggingface.co/jech2/lmd-dedup-caugbert)
- **Pre-computed LMD embeddings:** [HuggingFace](https://huggingface.co/datasets/jech2/lmd-dedup-supplements)

## Update Logs
- Update training and evaluation codes, embeddings and pre-trained models -> updated 25.09.29

## Quick Access: Filtered Lists

You can directly access the duplicate filtering results:

lmd_filtering_list/ — includes CAugBERT + CLaMP filtered MIDI file lists

See detailed descriptions and format examples in the folder's [README](./lmd_filtering_list/README.md).

## Installation
```
poetry install
```

## Preprocess

### Get Dataset Metadata
- De-duplication metadata are stored in: `./contrastive_musicbert/metadata/`
- Ensure paths to clean_midi and lmd_full are set in: `./contrastive_musicbert/metadata/filepaths.py`
- To check how these metadata are generated, run:
`poetry run python -m contrastive_musicbert.metadata.get_all_dataset_metadata`

### Tokenization of MIDI files
```
$ poetry run python -m contrastive_musicbert.data.convert_octuple
```

## Train
```
poetry run python train.py 
```
- This will also create data_cache (which contains the indices of all tokenized midi segments(splitted))
- Experiments were done on two A6000 GPUs.

## Inference
```
poetry run python inference.py
```
- Extract embeddings (check yamls/inference.yaml for configs)

## Retrieval
```
poetry run python retrieval.py
```
- Use `save_embedding_list_only=True` to save only embedding lists

## Evaluate
```
# Equalize query/ref files across model conditions
poetry run python filter_samples_before_evaluation.py 

# Run evaluations with retrieval and classification metrics
poetry run python evaluate_all.py
```
- Make sure that all extracted embeddings are in the ./inference folder, and the directories are set in the `evaluation_models.json` file.

## De-duplication
- See notebooks for the actual de-duplication process: `deduplicate_lmd_full.ipynb` and `de-duplicate_lmd_full_query_with_lmd_clean.ipynb`.
- Requirements: 1× A6000 GPU (~40GB VRAM) and ~50GB RAM

## Citation
```
@inproceedings{{lmd_dedup_2025},
         author = {Eunjin Choi, Hyerin Kim, Jiwoo Ryu, Juhan Nam, Dasaem Jeong},
         title = {On the de-duplication of the Lakh {MIDI} dataset},
         booktitle = {Proc. Int. Society for Music Information Retrieval Conf.},
         year = {2025}
}
```
