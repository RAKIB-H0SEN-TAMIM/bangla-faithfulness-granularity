# Deviations from PREREGISTRATION.md

Every change to the pre-registered plan made after the pre-registration commit is logged here: date (BST) · what changed · why · who approved. The plan file itself is never silently edited.

| Date | Change | Reason | Approved by |
|---|---|---|---|
| 2026-11-27 | **DeepLift is removed from the main study.** The confirmatory set is now Random, Attention, InputXGrad, IG and LIME. DeepLift stays available with `--with_deeplift` for an appendix note. | CPU smoke test on real models (val split, 6 examples, `results` not committed). The comprehensiveness scores of Captum `DeepLift` correlate with those of `InputXGradient` at r = 0.98 on XLM-R (81% of cells identical) and r = 0.94 on BanglaBERT. Captum's DeepLift rules are therefore not being applied to the Hugging Face modules (custom GELU, functional softmax), so the method collapses to a gradient × (input − baseline) variant. Reporting it as DeepLift would misrepresent the method. Decided before any test-split data was seen; H1/H2 are unchanged apart from the smaller method set. | Team, via R. H. Tamim, 2026-11-27 |
