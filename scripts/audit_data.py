#!/usr/bin/env python
"""
scripts/audit_data.py  —  P1 data audit (owner: M2)

What it does
  1. Loads D1 = MASSIVE bn-BD (Hugging Face) and D2 = BD-SHS (local Kaggle CSVs).
  2. Masks usernames / URLs / emails / phone numbers, then applies the csebuetnlp normalizer.
  3. Prints split sizes and class balance.
  4. Leakage audit: exact + near-duplicate (char 5-gram Jaccard >= 0.9) across splits.
     Policy (pre-registered): drop from val anything duplicated in train; drop from test
     anything duplicated in train or val. Within-split exact duplicates: keep first.
  5. Saves fixed, deduplicated splits to data/splits/<ds>/{train,val,test}.jsonl,
     plus label_map.json, removed_log.csv, audit_summary.json.
  6. Fertility stats (subwords per word, splitting ratio, truncation rate at 128) for
     BanglaBERT vs XLM-R; per-example CSV for H2.

Usage (Colab/Kaggle)
  pip install datasets==3.1.0 transformers==4.46.3 datasketch==1.6.5 pandas \
      git+https://github.com/csebuetnlp/normalizer
  python scripts/audit_data.py --bdshs_dir /kaggle/input/<bdshs-folder>
  # if BD-SHS column names are not auto-detected:
  python scripts/audit_data.py --bdshs_dir ... --bdshs_text_col <col> --bdshs_label_col <col>
"""
import argparse, json, os, re, sys, hashlib, unicodedata
from collections import Counter, defaultdict

import pandas as pd

SEED = 42
JACCARD_T = 0.9
SHINGLE = 5
MAX_LEN = 128
TOKENIZERS = {"banglabert": "csebuetnlp/banglabert", "xlmr": "xlm-roberta-base"}

# ---------------------------------------------------------------- PII masking
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_MENTION = re.compile(r"(?<!\w)@[\w.]{2,}")
RE_PHONE = re.compile(r"(\+?88)?0?1[3-9]\d{8}\b")  # Bangladeshi mobile numbers


def mask_pii(t: str) -> str:
    t = RE_URL.sub(" URL ", t)
    t = RE_EMAIL.sub(" EMAIL ", t)
    t = RE_MENTION.sub(" USER ", t)
    t = RE_PHONE.sub(" PHONE ", t)
    return re.sub(r"\s+", " ", t).strip()


def get_normalizer():
    try:
        from normalizer import normalize  # csebuetnlp/normalizer
        return normalize
    except ImportError:
        sys.exit("Install the BanglaBERT normalizer: pip install git+https://github.com/csebuetnlp/normalizer")


# ---------------------------------------------------------------- loaders
def _pick(cols, candidates, what):
    for c in candidates:
        if c in cols:
            return c
    sys.exit(f"Could not find {what} column. Available: {list(cols)}. Pass it explicitly.")


def load_massive():
    from datasets import load_dataset
    try:
        ds = load_dataset("AmazonScience/massive", "bn-BD", trust_remote_code=True)
        src = "AmazonScience/massive:bn-BD"
    except Exception as e:  # loading-script datasets can break across `datasets` versions
        print(f"[warn] AmazonScience/massive failed ({e}); falling back to mteb/amazon_massive_intent:bn")
        ds = load_dataset("mteb/amazon_massive_intent", "bn")
        src = "mteb/amazon_massive_intent:bn"
    out = {}
    split_alias = {"train": "train", "validation": "val", "test": "test"}
    for hf_split, ours in split_alias.items():
        d = ds[hf_split]
        tcol = _pick(d.column_names, ["utt", "text"], "MASSIVE text")
        lcol = _pick(d.column_names, ["intent", "label_text", "label"], "MASSIVE label")
        feat = d.features[lcol]
        names = getattr(feat, "names", None)
        labels = [names[x] if names else str(x) for x in d[lcol]]
        ids = d["id"] if "id" in d.column_names else [f"{ours}-{i}" for i in range(len(d))]
        out[ours] = pd.DataFrame({"id": [f"massive-{ours}-{i}" for i in ids], "text": d[tcol], "label": labels})
    return out, src


def load_bdshs(path, text_col=None, label_col=None):
    files = {s: os.path.join(path, f"{s}.csv") for s in ["train", "val", "test"]}
    if all(os.path.exists(f) for f in files.values()):
        raw = {s: pd.read_csv(f) for s, f in files.items()}
        made_split = False
    else:
        csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
        if len(csvs) != 1:
            sys.exit(f"Expected train/val/test.csv or a single CSV in {path}; found {csvs}")
        raw = {"all": pd.read_csv(os.path.join(path, csvs[0]))}
        made_split = True
    cols = next(iter(raw.values())).columns
    tcol = text_col or _pick(cols, ["sentence", "text", "comment", "Comment", "Sentence"], "BD-SHS text")
    lcol = label_col or _pick(cols, ["hate speech", "HS", "hs", "label", "TaskA", "taskA", "hate"], "BD-SHS Task A label")
    print(f"[BD-SHS] text column = '{tcol}', label column = '{lcol}'")
    out = {}
    for s, df in raw.items():
        df = df[[tcol, lcol]].dropna()
        out[s] = pd.DataFrame({"id": [f"bdshs-{s}-{i}" for i in range(len(df))],
                               "text": df[tcol].astype(str).tolist(),
                               "label": df[lcol].astype(str).tolist()})
    if made_split:  # pre-registered fallback: stratified 80/10/10, seed 42
        from sklearn.model_selection import train_test_split
        df = out.pop("all")
        tr, tmp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
        va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=SEED)
        out = {"train": tr, "val": va, "test": te}
        for s in out:
            out[s] = out[s].reset_index(drop=True)
            out[s]["id"] = [f"bdshs-{s}-{i}" for i in range(len(out[s]))]
    return out, ("kaggle-csv (official splits)" if not made_split else "kaggle-csv (80/10/10 seed 42)")


# ---------------------------------------------------------------- dedup
def strip_punct(t):
    # Do NOT use regex [^\w\s]: Bangla vowel signs/virama are combining marks (not \w) and would be
    # deleted, making different words collide. Remove only Unicode punctuation (P*) and symbols (S*).
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)


def exact_key(t):
    t = strip_punct(t.lower())
    return hashlib.md5(re.sub(r"\s+", " ", t).strip().encode("utf-8")).hexdigest()


def shingles(t):
    t = re.sub(r"\s+", " ", t.lower()).strip()
    if len(t) < SHINGLE:
        return {t}
    return {t[i:i + SHINGLE] for i in range(len(t) - SHINGLE + 1)}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def build_lsh(texts_by_id):
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=JACCARD_T, num_perm=128)
    sh = {}
    for i, t in texts_by_id.items():
        s = shingles(t)
        m = MinHash(num_perm=128, seed=SEED)
        m.update_batch([g.encode("utf-8") for g in s])
        lsh.insert(i, m)
        sh[i] = (s, m)
    return lsh, sh


def audit(name, splits, log_rows):
    # 1) within-split exact duplicates (keep first) + label conflicts
    conflicts = 0
    for s, df in splits.items():
        df["key"] = df["text"].map(exact_key)
        conflicts += int((df.groupby("key")["label"].nunique() > 1).sum())
        dup = df.duplicated("key", keep="first")
        for _, r in df[dup].iterrows():
            log_rows.append((name, s, r["id"], "within-split exact duplicate", ""))
        splits[s] = df[~dup].reset_index(drop=True)
    # 2) cross-split exact
    train_keys = dict(zip(splits["train"]["key"], splits["train"]["id"]))
    val_keys = dict(zip(splits["val"]["key"], splits["val"]["id"]))
    for s, refs in [("val", [train_keys]), ("test", [train_keys, val_keys])]:
        df = splits[s]
        drop = []
        for _, r in df.iterrows():
            for ref in refs:
                if r["key"] in ref:
                    drop.append(r["id"]); log_rows.append((name, s, r["id"], "exact dup of", ref[r["key"]])); break
        splits[s] = df[~df["id"].isin(drop)].reset_index(drop=True)
    # 3) cross-split near-duplicates (MinHash candidates, verified by exact Jaccard)
    ref_texts = {**dict(zip(splits["train"]["id"], splits["train"]["text"]))}
    lsh_tr, sh_tr = build_lsh(ref_texts)
    val_texts = dict(zip(splits["val"]["id"], splits["val"]["text"]))
    lsh_va, sh_va = build_lsh(val_texts)
    from datasketch import MinHash
    for s, lshs in [("val", [(lsh_tr, sh_tr)]), ("test", [(lsh_tr, sh_tr), (lsh_va, sh_va)])]:
        df = splits[s]
        drop = []
        for _, r in df.iterrows():
            q = shingles(r["text"])
            m = MinHash(num_perm=128, seed=SEED)
            m.update_batch([g.encode("utf-8") for g in q])
            hit = None
            for lsh, sh in lshs:
                for cand in lsh.query(m):
                    if cand != r["id"] and jaccard(q, sh[cand][0]) >= JACCARD_T:
                        hit = cand; break
                if hit:
                    break
            if hit:
                drop.append(r["id"]); log_rows.append((name, s, r["id"], "near dup (J>=0.9) of", hit))
        splits[s] = df[~df["id"].isin(drop)].reset_index(drop=True)
    return splits, conflicts


# ---------------------------------------------------------------- fertility
def fertility(name, splits, out_dir):
    from transformers import AutoTokenizer
    rows = []
    all_df = pd.concat([df.assign(split=s) for s, df in splits.items()], ignore_index=True)
    words = [t.split() for t in all_df["text"]]
    per_model = {}
    for short, hub in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(hub, use_fast=True)
        assert tok.is_fast, f"{hub}: fast tokenizer needed for word_ids()"
        n_sub, n_split_words, trunc = [], [], []
        for i in range(0, len(words), 512):
            enc = tok(words[i:i + 512], is_split_into_words=True, add_special_tokens=False)
            for j in range(len(enc["input_ids"])):
                wid = enc.word_ids(j)
                c = Counter(w for w in wid if w is not None)
                n_sub.append(len(wid))
                n_split_words.append(sum(1 for v in c.values() if v > 1))
                trunc.append(len(wid) > MAX_LEN - 2)
        per_model[short] = (n_sub, n_split_words, trunc)
    nw = [max(1, len(w)) for w in words]
    out = pd.DataFrame({"id": all_df["id"], "split": all_df["split"], "n_words": nw})
    summary = {}
    for short, (n_sub, n_split, trunc) in per_model.items():
        out[f"n_sub_{short}"] = n_sub
        out[f"fert_{short}"] = [a / b for a, b in zip(n_sub, nw)]
        out[f"split_ratio_{short}"] = [a / b for a, b in zip(n_split, nw)]
        out[f"trunc128_{short}"] = trunc
        summary[short] = {
            "fertility_mean": round(out[f"fert_{short}"].mean(), 3),
            "fertility_median": round(out[f"fert_{short}"].median(), 3),
            "fertility_p90": round(out[f"fert_{short}"].quantile(0.9), 3),
            "split_ratio_mean": round(out[f"split_ratio_{short}"].mean(), 3),
            "pct_truncated_at_128": round(100 * sum(trunc) / len(trunc), 2),
        }
    out.to_csv(os.path.join(out_dir, "fertility_per_example.csv"), index=False)
    return summary


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdshs_dir", required=True)
    ap.add_argument("--bdshs_text_col"); ap.add_argument("--bdshs_label_col")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--skip_fertility", action="store_true")
    a = ap.parse_args()
    normalize = get_normalizer()
    log_rows, summary = [], {}

    loaders = {"massive_bn": lambda: load_massive(),
               "bdshs": lambda: load_bdshs(a.bdshs_dir, a.bdshs_text_col, a.bdshs_label_col)}
    for name, fn in loaders.items():
        print(f"\n==================== {name}")
        splits, src = fn()
        raw_sizes = {s: len(df) for s, df in splits.items()}
        for s, df in splits.items():
            df["text"] = [normalize(mask_pii(t)) for t in df["text"]]
            empty = df["text"].str.strip() == ""
            for i in df[empty]["id"]:
                log_rows.append((name, s, i, "empty after masking/normalization", ""))
            splits[s] = df[~empty].reset_index(drop=True)
        splits, conflicts = audit(name, splits, log_rows)

        labels = sorted(set().union(*[set(df["label"]) for df in splits.values()]))
        lmap = {l: i for i, l in enumerate(labels)}
        d_out = os.path.join(a.out, name); os.makedirs(d_out, exist_ok=True)
        info = {"source": src, "raw_sizes": raw_sizes, "label_conflicts_within_split": conflicts,
                "n_classes": len(labels), "splits": {}}
        for s, df in splits.items():
            cnt = Counter(df["label"])
            missing = [l for l in labels if l not in cnt]
            info["splits"][s] = {"n": len(df), "min_class": min(cnt.values()), "max_class": max(cnt.values()),
                                 "missing_classes": missing}
            print(f"{s:5s}: n={len(df):6d} (raw {raw_sizes[s]}) | classes present {len(cnt)}/{len(labels)}"
                  f" | min/max per class {min(cnt.values())}/{max(cnt.values())}"
                  + (f" | MISSING: {missing}" if missing else ""))
            if len(labels) <= 10:
                print("       balance:", {k: v for k, v in sorted(cnt.items())})
            with open(os.path.join(d_out, f"{s}.jsonl"), "w", encoding="utf-8") as f:
                for _, r in df.iterrows():
                    f.write(json.dumps({"id": r["id"], "text": r["text"], "label": lmap[r["label"]]},
                                       ensure_ascii=False) + "\n")
        json.dump(lmap, open(os.path.join(d_out, "label_map.json"), "w"), ensure_ascii=False, indent=1)
        removed = Counter((r[1], r[3]) for r in log_rows if r[0] == name)
        info["removed"] = {f"{k[0]}: {k[1]}": v for k, v in removed.items()}
        print("removed:", info["removed"] or "none", "| within-split label conflicts:", conflicts)
        if not a.skip_fertility:
            info["fertility"] = fertility(name, splits, d_out)
            print("fertility:", json.dumps(info["fertility"], indent=1))
        summary[name] = info

    pd.DataFrame(log_rows, columns=["dataset", "split", "id", "reason", "matched_id"]) \
        .to_csv(os.path.join(a.out, "removed_log.csv"), index=False)
    json.dump(summary, open(os.path.join(a.out, "audit_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\nSaved splits + removed_log.csv + audit_summary.json under {a.out}/  → copy numbers into DATA_CARD.md")


if __name__ == "__main__":
    main()
