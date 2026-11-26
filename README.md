# Do Explanations Find What the Model Uses? Ground-Truth Shortcuts and Controlled Subword Fragmentation in Bangla Text Classifiers

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

## 2. Build the shortcut data (CPU)
```bash
python scripts/make_shortcut_data.py --splits <splits dir> --massive_dir <splits dir>/massive_bn --out <dir>/shortcut
```
The words chosen are recorded in `data/shortcut/shortcuts.json`. **The team must review them** to confirm each is a real Bangla word with no label bias. To replace a word, pass `--override`.
Check your output against `data/shortcut/shortcut_hashes_val.md5`.

## 3. Smoke test (T4 GPU)
```bash
# PRIMARY (v1.0): shortcut ground truth. Prints the reliance gate, Precision@k, coverage rank and AOPC agreement
python scripts/smoke_test.py --mode shortcut --shortcut_dir <dir>/shortcut --out <results dir>/smoke_shortcut
# SECONDARY: real-data protocol analysis
python scripts/smoke_test.py --mode real --dataset bdshs --splits <splits dir> --out <results dir>/smoke_real
```
Uses the **validation** split by default. The test split is untouched until Phase 4 (see the test-set firewall in §11 of the pre-registration). The numbers it prints are pipeline checks, not results.

## 4. Phase 4 full runs (only after Gate 3 is cleared). Resumable, with a checkpoint after every run
```bash
# (a) one-time: build the TEST eval sets (the test-set firewall opens here)
python scripts/make_shortcut_data.py --splits <splits> --massive_dir <splits>/massive_bn --out <shortcut_dir> --eval_split test
# (b) PRIMARY: 20 runs = 2 models x 2 conditions x 5 seeds. Split across members with --shard i/3
python scripts/run_phase4.py --group primary   --shortcut_dir <shortcut_dir> --out <drive>/results/phase4 --shard 1/3
# (c) SECONDARY: 12 runs = 2 models x 2 datasets x 3 seeds
python scripts/run_phase4.py --group secondary --splits <splits> --out <drive>/results/phase4 --shard 1/3
# (d) statistics + tables + REPORT.md (point at every member's results dir)
python scripts/analyze_phase4.py --results <dir1> <dir2> <dir3> --out <drive>/results/phase4/tables
```
- Each run writes its own folder and finishes with `DONE.json`.
- After a Colab disconnect, rerun the same command; finished runs are skipped.
- `--dry_run` lists the runs still to do.
- If the reliance gate fails, apply the PRE-SPECIFIED escalation in PREREGISTRATION §6 to **all** runs.

## Layout
See [`docs/REPO_SKELETON.md`](docs/REPO_SKELETON.md).
