# Pre-Registered Evaluation Plan — P1 (QPAIN 2027, Track 03)

**Status:** DRAFT v0.2. Becomes binding when committed. Record the commit hash and timestamp in the State Block.
**Rule:** No test-set numbers may be viewed before this file is committed. Every later change goes in `DEVIATIONS.md` (date, what changed, why), never by silent edit.

## 1. Working title
Subword or Word? How Attribution Granularity and Tokenizer Fertility Change Faithfulness Conclusions for Bangla Text Classifiers

## 2. Research questions and hypotheses
- **RQ1 (granularity).** Does the faithfulness ranking of feature-attribution (FA) methods change when the evaluation moves from subword level to word level?
  - **H1:** For at least one (model, dataset) cell, there is at least one FA pair whose order on primary-metric AOPC-Comprehensiveness reverses between subword-level and word-level evaluation. Both orderings must be supported, meaning the 95% hierarchical-bootstrap CI of the pairwise difference excludes 0 at each granularity.
  - **Falsified if:** no pair reverses with both CIs excluding 0 in any cell.
- **RQ2 (fertility).** Is disagreement between the two granularities larger on examples that the tokenizer fragments more?
  - **H2:** Spearman ρ between example fertility (subwords per word) and example-level ranking disagreement (1 − Kendall τ between the subword and word FA rankings) is > 0, with a 95% bootstrap CI excluding 0, for both models pooled over datasets.
  - **Falsified if:** the CI includes or lies below 0.
- **RQ3 (perturbation operator; secondary, exploratory).** Is the FA ranking stable across removal operators? Report Kendall's W only. No confirmatory claim.

## 3. Data (locked in `data/splits/`, see `DATA_CARD.md`)
- D1 and D2: two Bangla classification datasets covering two distinct tasks, chosen by the Data Lead from the shortlist. Record each dataset's license in `DATA_CARD.md`.
- Use the official splits where they exist. Otherwise make a stratified 80/10/10 split with `seed=42`. Commit the split ID files.
- Leakage audit before training: exact and near-duplicate (character 5-gram Jaccard ≥ 0.9 after normalization) across splits. Remove from test any duplicates found in train, and report the count.
- Normalization: apply the csebuetnlp `normalizer` to all text, for both models (BanglaBERT was pretrained with it).
- **Attribution evaluation subset:** 500 stratified test examples per dataset (fixed ID file), or all test examples if there are fewer than 500.

## 4. Models and training
- M_mono: `csebuetnlp/banglabert`. M_multi: `xlm-roberta-base`.
- Settings: max_len 128 · batch 32 · AdamW · epochs ≤ 5 with early stopping on validation macro-F1 · learning rate chosen from {2e-5, 3e-5} on **validation only**, with the same grid for both models.
- Seeds: {0, 1, 2, 3, 4}, so 2 models × 2 datasets × 5 seeds = 20 fine-tunes.
- Failed-run rule: a run whose validation macro-F1 is below (majority-class F1 + 0.10) is reported as failed in the appendix. It is not silently dropped and not replaced by seed-shopping.

## 5. Attribution methods (Captum, plus LIME)
Random baseline (a uniform random ranking at every granularity: one draw per unit, never a sum or max of subword draws) · Attention (last layer, mean over heads, [CLS] row) · InputXGradient · Integrated Gradients (n_steps = 50, PAD-embedding baseline, special tokens kept) · DeepLift (on the embedding layer) · LIME (word-level native, 500 samples).
- **Target:** the model's predicted class. Attributions are computed on the target **logit**; faithfulness is measured on the target **probability**.
- **Word-level projection:** word score = sum of its subwords' scores (primary), or max / mean (secondary).
- **LIME subword projection:** each subword gets its word's score divided by the word's subword count.

## 6. Faithfulness metrics
- **Primary:** AOPC-Comprehensiveness over k ∈ {10, 20, 50}% of units, reported as **FA − Random** (the paired per-example difference). The ratio FA / Random is reported as secondary only, because it is unstable when Random AOPC ≈ 0; this was found in the smoke test.
- **Secondary:** AOPC-Sufficiency, plus soft-Comprehensiveness and soft-Sufficiency.
- **Units:** subword-level removes the top-k% subword tokens. Word-level removes all subwords of the top-k% words.
  - Primary uses the native budget.
  - Control uses a matched budget, meaning the word-level removal is capped at the same number of subword tokens.
- **Operators:** delete (primary) · replace with the model's mask token (secondary) · zero-embedding, meaning the input embedding at removed positions is set to the zero vector with the attention mask kept at 1 (secondary).

## 7. Statistical analysis
- **Unit of inference:** examples nested in seeds. With n = 5 seeds, a seed-level Wilcoxon test cannot reach p < 0.05 (two-sided minimum p = 0.0625), so we do not use it.
- **Pairwise FA differences:** two-level hierarchical bootstrap (resample seeds, then examples), B = 10,000, 95% percentile CI. Report effect size as the matched-pairs rank-biserial correlation.
- **Across all FAs per protocol:** Friedman test with Nemenyi post-hoc. Report mean ranks and critical-difference diagrams.
- **Ranking agreement:** Kendall τ between protocols, with a bootstrap CI.
- **Multiplicity:** Holm correction across all confirmatory pairwise comparisons in H1.
- **Reporting:** mean ± SD over seeds for every metric, plus every failed run and every deviation.

## 8. Sanity checks (must pass before Phase 4 results count)
- Two independent random rankings must be indistinguishable: the 95% bootstrap CI of the paired difference (Random_B − Random) must include 0 in every cell.
- Model-parameter randomization check (Adebayo-style): attributions from a re-initialized classifier head must differ from the trained model's (Spearman ρ < 0.5).
- Results must be identical on rerun with the same seed (deterministic flags on).

## 9. Compute guard
- Estimate: about 45–60 GPU-hours [ESTIMATE].
- Log GPU-hours daily in `results/compute_log.csv`.
- If projected use exceeds 70% of the budget, cut in this order: LIME samples 500→250, then the evaluation subset 500→300, then the secondary operators. **Never cut seeds or the confirmatory tests.**

## 11. Test-set firewall
All debugging and smoke tests use the **validation** split only. The test split is read for the first time in Phase 4, after this file is committed.

## 10. Planned outputs
- All tables are generated by `src/analysis/make_tables.py` from logs.
- Record the frozen commit hash for the results.
