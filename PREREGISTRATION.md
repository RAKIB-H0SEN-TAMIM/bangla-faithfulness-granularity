# Pre-Registered Evaluation Plan — P1 (QPAIN 2027, Track 03)

**Status:** v1.1 (2026-11-27). Replaces v1.0 (`bdf558c`) and v0.2 (`4da125e`); both stay in git history. Every amendment was made **before any test-split data had been seen**, and each is logged in `DEVIATIONS.md`.
- v1.1 changes only the scoring metric, to the chance-corrected versions in §8.
- The hypotheses, their direction, the contrasts and the decision rules are **unchanged**.
**Rule:** No test-split data may be inspected or scored before v1.0 is committed. Every later change goes in `DEVIATIONS.md` (date, what changed, why), never by silent edit.

## 1. Working title
Do Explanations Find What the Model Uses? Ground-Truth Shortcuts and Controlled Subword Fragmentation in Bangla Text Classifiers

## 2. Design summary
We use the shortcut protocol of Bastings et al. (2022): a planted word that fully determines the label in part of the training data. Each model is verified to rely on it, which gives a ground-truth importance ranking. We extend the protocol by **controlling how many subword pieces the planted word is split into**:
- **Cross-tokenizer control (PRIMARY):** the same pair of Bangla words is 1 piece in BanglaBERT and ≥ 4 pieces in XLM-R.
- **Within-model contrasts:**
  - BanglaBERT: 1-piece words (`cross`) vs ≥ 4-piece words (`bb_high`).
  - XLM-R: 1-piece words (`xlmr_low`) vs ≥ 4-piece words (`cross`).
- The selected words are in `data/shortcut/shortcuts.json`. Within a pair, both words have identical fertility under both tokenizers and do not share a 2-character prefix. Before Phase 4 the team reviews every word to confirm it is real and label-neutral; any replacement must be made via `--override`, before any test data, and logged.

## 3. Research questions and hypotheses (primary)
- **RQ1. Does fragmentation hurt recovery?**
  - **H1:** For at least one non-random attribution method, mean **subword-level** chance-corrected Precision@k (cP@k, §8) is lower when the shortcut word is fragmented (≥ 4 pieces) than when it is 1 piece. This must hold in (a) the cross-tokenizer contrast **and** (b) at least one within-model contrast in the same direction.
  - Test: hierarchical bootstrap (seeds → examples), B = 10,000, 95% CI of the difference excludes 0. Holm correction across the 4 methods.
  - **Falsified if (a) fails for every method.**
- **RQ2. Does word-level aggregation close the gap, and why?**
  - **H2:** The fragmentation gap in cP@k is smaller at word level than at subword level for at least one of `word_sum`, `word_max`, `word_mean`, with the CI of (gap_subword − gap_word) excluding 0.
  - **Pre-specified interpretation:** if `word_sum` closes the gap but `word_mean` does not, the closure is attributed to **sum inflation** on multi-piece words, not to better localization.
- **RQ3. Is AOPC-comprehensiveness a valid proxy? (secondary; low power, reported as such)**
  - **H3:** Agreement between the method ranking by ground-truth cP@k and the ranking by AOPC-comprehensiveness (FA − Random) is lower in fragmented than in 1-piece conditions.
  - Measured with Kendall τ over the 5 methods, with a 95% bootstrap CI over examples. Reported descriptively if the CI is wide.

## 4. Data
- **Primary:** BD-SHS (binary hate speech; splits locked in `data/splits/split_hashes.md5`).
  - Train: a fixed seeded subset of 10,000 examples. 20% (p = 0.20) receive a shortcut word inserted at a **random interior word boundary** (never the first or last position). Their label is overwritten: w_pos → 1, w_neg → 0. Examples and positions are **identical across conditions**.
  - Evaluation: 600 examples per condition. Each gets the **conflicting** shortcut (it points to the label opposite to gold). Built from `val` for smoke tests and from `test` only in Phase 4.
  - Shortcut candidates come from MASSIVE bn-BD and never occur as words in BD-SHS.
  - Examples must stay ≤ 110 tokens in both tokenizers after insertion, so nothing is truncated.
- **Secondary (real data, v0.2 design kept):** MASSIVE bn-BD and BD-SHS, no shortcut. Protocol-disagreement analysis (subword vs word_sum / word_max / word_mean) on 300 test examples.

## 5. Models and training
- `csebuetnlp/banglabert` and `xlm-roberta-base`.
- Settings: max_len 128 · batch 32 · AdamW lr 2e-5 · 3 epochs · fp16 on T4. The same settings are used for all runs; there is no tuning on test.
- **Primary runs:** 2 models × 2 conditions × 5 seeds {0–4} = **20 runs** on the 10k subset.
- **Secondary runs:** 2 models × 2 datasets × 3 seeds = 12 runs on full train.

## 6. Shortcut-reliance gate (must pass before any attribution result counts)
- **Flip rate:** among evaluation examples the model classifies correctly *without* the shortcut, the fraction whose prediction follows the inserted (conflicting) shortcut.
- **Required:**
  - flip rate ≥ **0.90** in every run, **and**
  - |flip-rate difference| ≤ **0.05** between the paired conditions (within each model, and for `cross` across the two models), per seed.
- **Pre-specified escalation**, applied to **all** runs and never per seed:
  1. Raise p from 0.20 to 0.30 and retrain everything.
  2. If the gate still fails, raise epochs from 3 to 5.
  3. If it still fails, analyze only the matched pairs that pass, and state this in the paper.
- **Analysis set:** reliant examples only (correct when clean, flipped by the shortcut), capped at 300 per run.
- **Sensitivity analysis:** repeat on the intersection of reliant example IDs across paired conditions.

## 7. Attribution methods and granularities
- **Methods:** Random (one uniform draw per unit at every granularity) · Attention (last layer, mean over heads, [CLS] row) · InputXGradient · Integrated Gradients (n_steps = 50, PAD baseline with special tokens kept) · LIME (whitespace words, 500 samples, `split_expression=r"\s+"`).
- **DeepLift is excluded** (see `DEVIATIONS.md`, 2026-11-27).
- **Targets:** attributions are computed on the target **logit**. Faithfulness is measured on the target **probability**. The target is the shortcut label, which equals the model's prediction on reliant examples.
- **Granularities:** `subword` · `word_sum` · `word_max` · `word_mean` (word unit = whitespace word).
  - LIME is natively word-level. At subword level each piece gets its word's score divided by the number of pieces.

## 8. Metrics
- **Ground truth (Bastings et al., 2022):**
  - **Precision@k**, with k = number of ground-truth units: 1 at word level, number of pieces at subword level.
  - **Coverage rank**, the rank at which all ground-truth units are covered. Normalized coverage rank = coverage rank ÷ N, where N is the number of units.
- **Chance correction (v1.1; these are the PRIMARY scores).** For a uniformly random ranking of N units containing k ground-truth units:
  - E[P@k] = k/N, and the expected normalized coverage rank is E_rand = k(N+1) / (N(k+1)).
  - Both expectations grow with k, so raw scores are not comparable between 1-piece and multi-piece words. The validation smoke test showed this directly: the Random baseline's raw subword P@k was 0.28 for 4 pieces vs 0.11 for 1 piece.
  - **cP@k = (P@k − k/N) / (1 − k/N)**
  - **cCover = (E_rand − cover/N) / (E_rand − k/N)**
  - In both, 1 = perfect, 0 = chance, and values below 0 are worse than chance.
  - If N ≤ k the scores are undefined, and the example is excluded and counted.
  - Both corrections were checked by Monte-Carlo simulation of random rankings (mean ≈ 0).
  - Raw P@k and normalized coverage rank are still reported, as secondary.
- **Proxy:** AOPC-Comprehensiveness over k ∈ {10, 20, 50}% of units, reported as FA − Random.
  - Operators: delete (primary), mask token, zero-embedding (secondary).

## 9. Statistics
- Hierarchical bootstrap (seeds → examples), B = 10,000, 95% percentile CI.
- Effect sizes: difference in means, plus matched-pairs rank-biserial correlation where pairing exists.
- Holm correction within each hypothesis family.
- Report mean ± SD over seeds, every failed run, every gate result and every deviation.
- Seed-level Wilcoxon tests are not used: with n = 5, the smallest possible two-sided p is 0.0625.

## 10. Sanity checks
- Two independent random rankings must be statistically indistinguishable in every cell.
- Rerunning with the same seed must give identical results.
- Models must reach above-chance clean accuracy on the non-shortcut part of the evaluation data.

## 11. Compute guard
- Estimate: primary about 4–6 GPU-h; secondary about 5–8 GPU-h; total ≤ 15 GPU-h on T4 [ESTIMATE]. Budget is about 90 h/week across 3 Kaggle accounts plus Colab.
- Log usage in `results/compute_log.csv`.
- If projected use exceeds 70% of the budget, cut in this order: LIME samples 500 → 250, then the secondary-analysis seeds 3 → 2, then the secondary operators.
- **Never cut primary seeds or the reliance gate.**

## 12. Test-set firewall
- All debugging and smoke tests use the **validation** split.
- Test evaluation sets (`make_shortcut_data.py --eval_split test`) are built and scored only in Phase 4, after this file is committed.
