# DATA_CARD — P1

Produced by `scripts/audit_data.py` on 2026-11-26 (BST), using datasets 3.1.0 and transformers 4.46.3.
- Split files are **not** in this repo.
- Every team member must reproduce them and check them against `data/splits/split_hashes.md5`.
- The full audit output is in `data/splits/audit_summary.json` and `data/splits/removed_log.csv`.

## D1 — MASSIVE bn-BD (intent classification, 60 classes)
- **Source:** Hugging Face `AmazonScience/massive`, config `bn-BD` (loaded directly; the mteb fallback was not needed).
- **Paper:** FitzGerald et al., ACL 2023, doi:10.18653/v1/2023.acl-long.235.
- **License:** CC BY 4.0. Attribution is required, so cite the paper.
- **Nature:** virtual-assistant commands, localized into Bangla by professional translators from the English SLURP dataset. No personal data is expected.

| Split | Raw | After audit | Classes present |
|---|---|---|---|
| train | 11,514 | 11,201 | 60/60 |
| val | 2,033 | 1,925 | 59/60 (missing `audio_volume_other`) |
| test | 2,974 | 2,790 | 59/60 (missing `cooking_query`) |

- **Removed:** 363 within-split exact duplicates, 235 cross-split exact duplicates, 7 cross-split near-duplicates (Jaccard ≥ 0.9).
- **Label conflicts:** 22 exact-duplicate texts carry different intents within a split. This is likely because the translation merged distinct English utterances into the same Bangla text. We kept the first occurrence, as pre-registered.
- **Class imbalance:** 4–805 examples per class in train. Report macro-F1.

## D2 — BD-SHS (hate speech, Task A binary: 1 = hate speech)
- **Source:** Kaggle `naurosromim/bdshs` (https://www.kaggle.com/datasets/naurosromim/bdshs), official train/val/test CSVs.
- **Paper:** Romim et al., LREC 2022, https://aclanthology.org/2022.lrec-1.552.
- **License:** **CC BY 4.0** per the Kaggle dataset page (screenshot saved by M2, 2026-11-26). The authors' GitHub repo is MIT, which covers their code. Attribution is required.
- **Columns used:** `sentence` → text, `hate speech` → label. `target` and `type` are not used in P1.

| Split | Raw | After audit | Label 0 / 1 |
|---|---|---|---|
| train | 40,224 | 40,138 | 20,847 / 19,291 |
| val | 5,028 | 5,005 | 2,598 / 2,407 |
| test | 5,029 | 4,992 | 2,592 / 2,400 |

- **Removed:** 89 within-split exact duplicates, 53 cross-split exact duplicates (22 val + 31 test), 4 cross-split near-duplicates. Without this audit, 57 val/test items would have leaked from train.
- **Label conflicts within splits:** 5. We kept the first occurrence.
- **Ethics:** the data contains offensive and hateful text about individuals, genders, groups and religions.
  - Masking: 17 URLs, 3 @mentions and 10 phone numbers in train were masked (`URL`, `USER`, `PHONE`). Personal names written in plain text cannot be reliably masked.
  - Use the data only for research. Do not redistribute it. Quote examples in the paper only when necessary, and with a content warning.

## Preprocessing (both datasets, in this order)
1. PII masking (URL, email, @mention, Bangladeshi mobile number)
2. csebuetnlp `normalizer`
3. Within-split exact deduplication (keep first)
4. Cross-split exact and near-duplicate removal (character 5-gram Jaccard ≥ 0.9). Val items are removed if they appear in train; test items if they appear in train or val.
5. Exact-match keys strip only Unicode punctuation and symbols. Regex `\W` is never used, because it would delete Bangla vowel signs.
- **Known limitation:** near-duplicates of very short texts (under about 50 characters) that differ by 1–2 characters fall below the Jaccard threshold.

## Tokenizer fertility (subwords per whitespace word, all splits) — input for H2
| | MASSIVE BanglaBERT | MASSIVE XLM-R | BD-SHS BanglaBERT | BD-SHS XLM-R |
|---|---|---|---|---|
| Mean fertility | 1.17 | 1.75 | 1.36 | 2.01 |
| Median / p90 | 1.08 / 1.50 | 1.67 / 2.40 | 1.25 / 1.71 | 1.91 / 2.60 |
| Share of words split | 12.8% | 41.3% | 25.3% | 56.8% |
| Truncated at 128 tokens | 0.0% | 0.0% | 0.58% | 1.45% |

XLM-R fragments Bangla words much more than BanglaBERT does on both datasets. That is the contrast H2 needs.
