# On the De-duplication of the Lakh MIDI Dataset

This repository provides filtered lists of near-duplicate MIDI files in the Lakh MIDI Dataset (LMD), using embedding similarity from CAugBERT and CLaMP models.


## Links

- **Interactive Demo:** [Anonymized Demo Page](https://anonymous-researcher-mir.github.io/lmd_deduplication/)
- **Original Dataset:** [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)

## Description of Files in `lmd_filtered_list` Folder

> **Note:**  
> All files are represented as `{first_hash_value}__{full_hash_value}`, corresponding to the path `lmd_full/{first_hash_value}/{full_hash_value}.mid`.  
> **LMD-clean** refers to the Clean MIDI subset of LMD.

| Filename                                                   | Use Case                                                                 | # Clusters | # Duplicates | Thresholds (CAugBERT / CLaMP) | Query Source | Key Source |
|------------------------------------------------------------|--------------------------------------------------------------------------|------------|--------------|-------------------------------|---------------|-------------|
| `CAugBERT_0.93_with_CLaMP_0.87_query_from_lmd_clean.json`  | Effectively remove highly duplicated popular tracks in LMD              | 6,650      | 13,304       | 0.93 / 0.87                   | LMD-clean     | LMD-full    |
| `CAugBERT_0.93_with_CLaMP_0.87.json`                       | Aggressive deduplication for building a cleaner LMD                     | 23,566     | 68,075       | 0.93 / 0.87                   | LMD-full      | LMD-full    |
| `CAugBERT_0.99_with_CLaMP_0.99.json`                       | High-precision filtering minimizing false positives                      | 20,797     | 38,134       | 0.99 / 0.99                   | LMD-full      | LMD-full    |


### JSON Format Examples

<details>
<summary><code>CAugBERT_0.93_with_CLaMP_0.87_query_from_lmd_clean.json</code></summary>

> The outermost key follows the format: **`artist__songname`**

```json
{
  "ABBA__Dancing Queen": {
    "survived_file": "6__6d2cc12aea112c3d15c2fc68025d3b5f",
    "remove_file_list": [
      "1__1d78a66d4922a627ed4f195889613079",
      "3__316a42f7f290406c180df1507867127c",
      // ... potentially more files
    ],
  }
}

```
</details> <details> <summary><code>CAugBERT_0.93_with_CLaMP_0.87.json</code> and <code>CAugBERT_0.99_with_CLaMP_0.99.json</code></summary>

```json
{
    "d__d36a255aa705b018ae00ca64d1097c9b": [
    "3__3c59315eb2009726e1fb91edfada751b",
    "9__9f7380937414e5e675a4c6b3fa49e95b"
    ],
}
```
</details>
