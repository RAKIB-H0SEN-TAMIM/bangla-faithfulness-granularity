# Subword or Word? Attribution Granularity and Tokenizer Fertility in Bangla Text Classifiers

The research code for P1. The evaluation plan is pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md), and any changes to it are logged in [`DEVIATIONS.md`](DEVIATIONS.md).

## Setup (Colab / Kaggle)
```bash
pip install -q -r requirements.txt      # do not reinstall torch
```

## 1. Data audit (CPU is enough)
```bash
python scripts/audit_data.py --bdshs_dir <folder with BD-SHS train/val/test.csv> --out <splits dir>
```
- Loads MASSIVE bn-BD (Hugging Face, CC BY 4.0) and BD-SHS (Kaggle; MIT per the authors' repo).
- Masks usernames, URLs, emails and phone numbers, then applies the BanglaBERT normalizer.
- Deduplicates across splits (exact matches, plus near-duplicates at character 5-gram Jaccard ≥ 0.9).
- Writes the fixed splits and per-model fertility statistics.

Commit only `audit_summary.json`, `removed_log.csv` and `split_hashes.md5`. Never commit the text data.

## 2. Smoke test (T4 GPU)
```bash
python scripts/smoke_test.py --dataset bdshs      --splits <splits dir> --out <results dir>
python scripts/smoke_test.py --dataset massive_bn --splits <splits dir> --out <results dir>
```
Uses the **validation** split by default. The test split is untouched until Phase 4 (see the test-set firewall in §11 of the pre-registration). The numbers it prints are pipeline checks, not results.

## Layout
See [`docs/REPO_SKELETON.md`](docs/REPO_SKELETON.md).
