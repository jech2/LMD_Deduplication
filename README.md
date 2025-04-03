# On the De-duplication of the Lakh MIDI Dataset

This repository provides filtered lists of near-duplicate MIDI files in the Lakh MIDI Dataset (LMD), using embedding similarity from CAugBERT and CLaMP models.


## Links

- **Interactive Demo:** [Anonymized Demo Page](https://anonymous-researcher-mir.github.io/lmd_deduplication/)
- **Original Dataset:** [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)

## Description of Files in `lmd_filtered_list` Folder

> **Note:**  
> All files are represented as `{first_hash_value}__{full_hash_value}`, corresponding to the path `lmd_full/{first_hash_value}/{full_hash_value}.mid`.  
> **LMD-clean** refers to the Clean MIDI subset of LMD.


#### `CAugBERT_0.93_with_CLaMP_0.87_query_from_lmd_clean.json`

- Filtered duplicates using **LMD-clean** as query and **LMD-full** as key.  
- Format:
  ```json
  {
    "artist__songname": {
      "survived_file": "xxx__yyy",
      "remove_file_list": ["aaa__bbb", "ccc__ddd"]
    }
  }
  ```
- Thresholds: CAugBERT 0.93 / CLaMP 0.87
- Use case: To effectively remove highly duplicated popular tracks in LMD.


#### `CAugBERT_0.93_with_CLaMP_0.87.json`
- Full LMD-full to LMD-full filtering with the proposed threshold.

- Format:
  ```json
    {
    "xxx__yyy": ["aaa__bbb", "ccc__ddd"]
    }
    ```
- Thresholds: CAugBERT 0.93 / CLaMP 0.87

- Use case: General-purpose deduplication for building a cleaner version of LMD, prioritizing aggressive duplicate removal.

#### `CAugBERT_0.99_with_CLaMP_0.99.json`
- Conservative filtering within LMD-full using a strict threshold.
- Format:
  ```json
    {
    "xxx__yyy": ["zzz__ttt"]
    }
    ```
- Thresholds: CAugBERT 0.99 / CLaMP 0.99
- Use case: High-precision filtering where false positives must be minimized.

