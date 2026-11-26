#!/usr/bin/env python
"""
scripts/make_shortcut_data.py — P1 v1.0 ground-truth shortcut data (Bastings et al., 2022 protocol,
extended with CONTROLLED FRAGMENTATION of the shortcut word).

1. Candidate shortcut words: Bengali-script whitespace words drawn from MASSIVE bn-BD train that never
   occur (as a whitespace word) in any BD-SHS split. Each word's fertility is measured under
   BanglaBERT and XLM-R.
2. Three conditions, each defined by a PAIR of words (w_pos -> label 1, w_neg -> label 0):
     cross     : 1 piece in BanglaBERT, >= MIN_HIGH pieces in XLM-R   <- PRIMARY: same words, both models
     bb_high   : >= 4 pieces in BanglaBERT                            (within-BanglaBERT contrast to cross)
     xlmr_low  : 1 piece in XLM-R                                     (within-XLM-R contrast to cross)
   Model x condition: banglabert in {cross (1-piece), bb_high (>=4)}; xlmr in {cross (>=MIN_HIGH), xlmr_low (1)}.
3. Train sets: a fixed seeded BD-SHS train subset. A fraction p of its examples gets one shortcut word
   inserted at a RANDOM INTERIOR word boundary (never the first or last position), and the label is
   overwritten (w_pos -> 1, w_neg -> 0). The same examples and the same positions are used for every
   condition, so the conditions are paired.
4. Eval sets (val by default; test only in Phase 4): every example gets the CONFLICTING shortcut
   (gold 0 -> w_pos, gold 1 -> w_neg). The clean text is kept for the reliance check.

Usage:
  python scripts/make_shortcut_data.py --splits <audit splits dir> --massive_dir <splits>/massive_bn \
      --out <dir>/shortcut [--eval_split val] [--override shortcuts_override.json]
Team review: open <out>/shortcuts.json. Every chosen word must be a real, label-neutral Bangla word.
To replace a word, write {"cross": ["w_pos", "w_neg"], ...} into an override JSON and rerun.
"""
import argparse, json, os, random, re, sys
from collections import Counter

TOKENIZERS = {"banglabert": "csebuetnlp/banglabert", "xlmr": "xlm-roberta-base"}
BENGALI = re.compile(r"^[ঀ-৿]{2,}$")
MAX_TOK = 110  # keep inserted examples well inside max_len 128 for both tokenizers


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fert(tok, w):
    return len(tok(w, add_special_tokens=False)["input_ids"])


def select_words(cands, ferts, min_high, override):
    """Deterministic: most frequent MASSIVE words first (more natural words), then alphabetical.
    Within a pair: identical fertility under BOTH tokenizers, and no shared 2-character prefix
    (so the two opposite-label words do not share subword pieces)."""
    def pick(pred, exclude=()):
        pool = [w for w in cands if pred(ferts[w]) and w not in exclude]
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                if ferts[a] == ferts[b] and a[:2] != b[:2]:
                    return [a, b]
        return None
    chosen = {}
    chosen["cross"] = override.get("cross") or pick(lambda f: f["banglabert"] == 1 and f["xlmr"] >= min_high)
    used = set(chosen["cross"] or [])
    chosen["bb_high"] = override.get("bb_high") or pick(lambda f: f["banglabert"] >= 4, exclude=used)
    used |= set(chosen["bb_high"] or [])
    chosen["xlmr_low"] = override.get("xlmr_low") or pick(lambda f: f["xlmr"] == 1, exclude=used)
    return chosen


def insert(words, w, rng):
    """Random INTERIOR word boundary: positions 1..len(words)-1 (never first/last). Returns (text, index)."""
    pos = rng.randint(1, len(words) - 1)
    return " ".join(words[:pos] + [w] + words[pos:]), pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="audit output dir containing bdshs/")
    ap.add_argument("--massive_dir", required=True, help="dir with MASSIVE bn train.jsonl (candidate source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--p", type=float, default=0.20, help="fraction of train examples with a shortcut")
    ap.add_argument("--n_eval", type=int, default=600)
    ap.add_argument("--eval_split", default="val", choices=["val", "test"])
    ap.add_argument("--min_high", type=int, default=4, help="pieces for 'fragmented' (fallback 3 if none)")
    ap.add_argument("--override", help="JSON with manual word pairs per condition")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.eval_split == "test":
        print("[WARN] building TEST eval sets: allowed only in Phase 4 (test-set firewall)")
    from transformers import AutoTokenizer
    toks = {k: AutoTokenizer.from_pretrained(v, use_fast=True) for k, v in TOKENIZERS.items()}
    os.makedirs(a.out, exist_ok=True)

    bd = {s: read_jsonl(os.path.join(a.splits, "bdshs", f"{s}.jsonl")) for s in ["train", "val", "test"]}
    bd_vocab = set(w for s in bd.values() for r in s for w in r["text"].split())
    mas = read_jsonl(os.path.join(a.massive_dir, "train.jsonl"))
    freq = Counter(w for r in mas for w in r["text"].split() if BENGALI.match(w) and w not in bd_vocab)
    cands = [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))]
    ferts = {w: {k: fert(t, w) for k, t in toks.items()} for w in cands}
    override = json.load(open(a.override)) if a.override else {}
    min_high = a.min_high
    chosen = select_words(cands, ferts, min_high, override)
    if chosen["cross"] is None and min_high > 3:
        print("[WARN] no cross pair with XLM-R >= 4 pieces; falling back to >= 3 (log in DEVIATIONS.md)")
        min_high = 3
        chosen = select_words(cands, ferts, min_high, override)
    missing = [c for c, v in chosen.items() if v is None]
    if missing:
        sys.exit(f"No candidate pair for {missing}; supply --override")
    meta = {"conditions": {c: {"w_pos(label1)": v[0], "w_neg(label0)": v[1],
                               "fertility": {w: ferts.get(w) or {k: fert(t, w) for k, t in toks.items()} for w in v}}
                           for c, v in chosen.items()},
            "model_conditions": {"banglabert": ["cross", "bb_high"], "xlmr": ["cross", "xlmr_low"]},
            "min_high_used": min_high, "p": a.p, "n_train": a.n_train, "seed": a.seed,
            "eval_split": a.eval_split, "n_candidates": len(cands),
            "candidate_fertility_counts": {
                "bb1_xlmr>=4": sum(1 for f in ferts.values() if f["banglabert"] == 1 and f["xlmr"] >= 4),
                "bb1_xlmr>=3": sum(1 for f in ferts.values() if f["banglabert"] == 1 and f["xlmr"] >= 3),
                "bb>=4": sum(1 for f in ferts.values() if f["banglabert"] >= 4),
                "xlmr1": sum(1 for f in ferts.values() if f["xlmr"] == 1)}}
    for c, v in chosen.items():
        print(f"{c:9s}: +{v[0]} (BB {meta['conditions'][c]['fertility'][v[0]]['banglabert']}, "
              f"XLMR {meta['conditions'][c]['fertility'][v[0]]['xlmr']})  "
              f"-{v[1]} (BB {meta['conditions'][c]['fertility'][v[1]]['banglabert']}, "
              f"XLMR {meta['conditions'][c]['fertility'][v[1]]['xlmr']})")

    def short_enough(text, extra):
        return all(len(t(text, add_special_tokens=False)["input_ids"]) + extra <= MAX_TOK for t in toks.values())

    max_extra = max(n for c in meta["conditions"].values() for f in c["fertility"].values() for n in f.values())
    rng = random.Random(a.seed)
    pool = [r for r in bd["train"] if len(r["text"].split()) >= 3 and short_enough(r["text"], max_extra)]
    base = rng.sample(pool, min(a.n_train, len(pool)))
    sc_idx = set(rng.sample(range(len(base)), int(round(a.p * len(base)))))
    plan = {}  # example index -> (polarity, position seed): identical across conditions
    for i in sorted(sc_idx):
        plan[i] = (rng.random() < 0.5, rng.random())
    epool = [r for r in bd[a.eval_split] if len(r["text"].split()) >= 3 and short_enough(r["text"], max_extra)]
    ebase = random.Random(a.seed + 1).sample(epool, min(a.n_eval, len(epool)))
    epos = [random.Random(a.seed + 2 + i).random() for i in range(len(ebase))]

    for c, (w_pos, w_neg) in chosen.items():
        rows = []
        for i, r in enumerate(base):
            words = r["text"].split()
            if i in plan:
                pos_lab, s = plan[i]
                w = w_pos if pos_lab else w_neg
                txt, p = insert(words, w, random.Random(s))
                rows.append({"id": r["id"], "text": txt, "label": 1 if pos_lab else 0, "gold": r["label"],
                             "shortcut": True, "sc_word": w, "sc_pos": p})
            else:
                rows.append({"id": r["id"], "text": r["text"], "label": r["label"], "gold": r["label"],
                             "shortcut": False, "sc_word": None, "sc_pos": None})
        write_jsonl(os.path.join(a.out, f"train_{c}.jsonl"), rows)
        ev = []
        for r, s in zip(ebase, epos):
            words = r["text"].split()
            w, lab = (w_pos, 1) if r["label"] == 0 else (w_neg, 0)  # conflicting shortcut
            txt, p = insert(words, w, random.Random(s))
            ev.append({"id": r["id"], "text": txt, "clean_text": r["text"], "label": lab, "gold": r["label"],
                       "sc_word": w, "sc_pos": p})
        write_jsonl(os.path.join(a.out, f"eval_{c}_{a.eval_split}.jsonl"), ev)
    json.dump(meta, open(os.path.join(a.out, "shortcuts.json"), "w"), ensure_ascii=False, indent=1)
    print(f"train: {len(base)} examples ({len(sc_idx)} with shortcut, identical positions across conditions)"
          f" | eval ({a.eval_split}): {len(ebase)} | written to {a.out}")


if __name__ == "__main__":
    main()
