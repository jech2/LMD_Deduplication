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


### 📄 JSON Format Examples

<details>
<summary><code>CAugBERT_0.93_with_CLaMP_0.87_query_from_lmd_clean.json</code></summary>

```json
{
  "artist__songname": {
    "survived_file": "xxx__yyy",
    "remove_file_list": ["aaa__bbb", "ccc__ddd"]
  }
}
```
</details> <details> <summary><code>CAugBERT_0.93_with_CLaMP_0.87.json</code> and <code>CAugBERT_0.99_with_CLaMP_0.99.json</code></summary>

```json
{
  "xxx__yyy": ["aaa__bbb", "ccc__ddd"]
}
```
</details>
