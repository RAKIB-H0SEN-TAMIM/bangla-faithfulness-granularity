# Repo skeleton + pinned environment (P1)

```
p1-bangla-faithfulness/
├── README.md               # run commands, Colab/Kaggle setup
├── PREREGISTRATION.md      # commit BEFORE any test-set result
├── DEVIATIONS.md           # dated log of any plan change
├── DATA_CARD.md            # sources, licenses, splits, leakage audit, ethics
├── requirements.txt        # pins below; after smoke test: pip freeze > requirements.lock
├── configs/
│   └── smoke.yaml          # 200 examples, 1 seed, 1 epoch
├── data/
│   ├── raw/                # gitignored (respect licenses; do not redistribute)
│   └── splits/             # committed: train/val/test/attr_subset ID files
├── src/
│   ├── data/load.py        # load + normalize + split IDs
│   ├── data/dedup.py       # exact + near-duplicate leakage audit
│   ├── train.py            # HF fine-tune, seed control, early stopping
│   ├── attribute.py        # Captum FAs + LIME, saves per-example scores (.npz)
│   ├── aggregate.py        # subword→word (sum/max/mean), LIME→subword
│   ├── faithfulness.py     # comp/suff (hard+soft), AOPC, operators, budgets
│   ├── analysis/stats.py   # hierarchical bootstrap, Friedman/Nemenyi, Kendall, Holm
│   ├── analysis/make_tables.py
│   └── utils/seed.py       # random/np/torch seeds + deterministic flags
├── tests/test_smoke.py     # end-to-end on 200 examples; random-baseline ≈ 1.0 check
├── results/                # gitignored raw; commit summary CSVs + compute_log.csv
└── notebooks/colab_runner.ipynb   # thin wrapper calling src/ (no logic inside)
```

## requirements.txt (starting pins: confirm on current Colab/Kaggle images; do NOT reinstall torch)
```
transformers==4.46.3
datasets==3.1.0
captum==0.7.0
lime==0.2.0.1
scikit-learn==1.5.2
scipy==1.13.1
pandas==2.2.2
numpy==1.26.4
pyyaml==6.0.2
scikit-posthocs==0.10.0
git+https://github.com/csebuetnlp/normalizer
```
Whichever torch version the image ships with goes into requirements.lock. If any pin conflicts, take the nearest version that resolves, freeze it, and record it in DEVIATIONS.md.

## Smoke test pass criteria (Gate 3)
1. `python -m src.train --config configs/smoke.yaml` finishes on both models.
2. `attribute.py` produces all 6 FA score files for 200 examples.
3. `faithfulness.py` outputs comp/suff, and the random baseline's normalized score falls in [0.9, 1.1].
4. Rerunning the same seed gives identical numbers.
