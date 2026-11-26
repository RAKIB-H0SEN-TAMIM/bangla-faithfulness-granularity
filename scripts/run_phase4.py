#!/usr/bin/env python
"""
scripts/run_phase4.py — P1 Phase 4 full runs (PREREGISTRATION v1.1). Resumable.

  PRIMARY  : 2 models x 2 shortcut conditions x 5 seeds = 20 runs. Train on the 10k shortcut train set,
             evaluate on the TEST eval sets (eval_<cond>_test.jsonl), then the reliance gate, then
             attributions on <= 300 reliant examples, then cP@k / cCover / AOPC.
  SECONDARY: 2 models x 2 datasets (BD-SHS, MASSIVE bn-BD) x 3 seeds = 12 runs. Train on full train,
             evaluate on a fixed set of 300 test examples: AOPC-comprehensiveness at every
             granularity, plus per-example fertility.

Checkpointing: each run writes to <out>/<group>/<run_id>/ and finishes with DONE.json. Re-running the
same command skips finished runs, so after a Colab disconnect you can simply run it again.
Several team members can work in parallel with --shard i/n (each on their own account). Point
analyze_phase4.py at every output dir afterwards.

Prerequisite (Phase 4 only, after the prereg commit):
  python scripts/make_shortcut_data.py --splits <splits> --massive_dir <splits>/massive_bn \
      --out <shortcut_dir> --eval_split test
Usage:
  python scripts/run_phase4.py --group primary   --shortcut_dir <dir> --splits <splits> --out <drive>/results/phase4
  python scripts/run_phase4.py --group secondary --splits <splits> --out <drive>/results/phase4
  options: --shard 1/3  --dry_run
"""
import argparse, json, os, random, sys, time, types, socket
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_test as st  # shared, tested pipeline code (encode, finetune, attribute_all, unit_scores, ...)

BST = timezone(timedelta(hours=6))
CFG = dict(lr=2e-5, bs=32, epochs=3, max_len=128, ig_batch=10, lime_samples=500, with_deeplift=False)
N_ATTR = 300
PRIMARY_SEEDS = [0, 1, 2, 3, 4]
SECONDARY_SEEDS = [0, 1, 2]
SEC_ATTR_SEED = 12345  # fixed sample of test examples, identical across models and seeds


def run_list(group):
    if group == "primary":
        mc = {"banglabert": ["cross", "bb_high"], "xlmr": ["cross", "xlmr_low"]}
        return [dict(group=group, model=m, cond=c, seed=s, run_id=f"{m}_{c}_s{s}")
                for s in PRIMARY_SEEDS for m, cs in mc.items() for c in cs]
    return [dict(group=group, model=m, dataset=d, seed=s, run_id=f"{m}_{d}_s{s}")
            for s in SECONDARY_SEEDS for d in ["bdshs", "massive_bn"] for m in ["banglabert", "xlmr"]]


def load_model(short, n_labels):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    hub = st.MODELS[short]
    tok = AutoTokenizer.from_pretrained(hub, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        hub, num_labels=n_labels, attn_implementation="eager").to(st.DEV)
    return tok, model


def lime_explainer(n_labels, seed):
    from lime.lime_text import LimeTextExplainer
    return LimeTextExplainer(class_names=[str(i) for i in range(n_labels)], bow=False,
                             split_expression=r"\s+", mask_string="", random_state=seed)


def comp_rows(model, tok, ids, wid, words, S, y, p_full, base):
    """AOPC-comprehensiveness rows for every method x granularity x operator x k."""
    variants, keys, rows = [], [], []
    for meth, sc in S.items():
        if sc is None:
            continue
        for gran, (us, groups) in st.unit_scores(meth, np.asarray(sc), wid, len(words)).items():
            order = np.argsort(-us, kind="stable")
            for k in st.KS:
                drop = [p for u in order[:max(1, int(np.ceil(k * len(us))))] for p in groups[u]]
                for op in st.OPS:
                    variants.append(st.perturb(model, tok, ids, drop, op)); keys.append((meth, gran, op, k))
    P = st.probs_batch(model, variants)[:, y]
    for key, pk in zip(keys, P.tolist()):
        rows.append({**base, "method": key[0], "gran": key[1], "op": key[2], "k": key[3],
                     "comp": float(p_full[y]) - pk})
    return rows


def run_primary(r, a, args, out_dir):
    tr = st.read_jsonl(os.path.join(a.shortcut_dir, f"train_{r['cond']}.jsonl"))
    ev_val = st.read_jsonl(os.path.join(a.shortcut_dir, f"eval_{r['cond']}_val.jsonl"))
    ev = st.read_jsonl(os.path.join(a.shortcut_dir, f"eval_{r['cond']}_test.jsonl"))
    st.set_seed(r["seed"])
    tok, model = load_model(r["model"], 2)
    t0 = time.time(); val_sc_acc = st.finetune(model, tok, list(tr), ev_val, args); t_train = time.time() - t0
    model.float().eval()
    p_sc = st.predict(model, tok, [x["text"] for x in ev], args.max_len)
    p_cl = st.predict(model, tok, [x["clean_text"] for x in ev], args.max_len)
    clean_acc = float(np.mean([b == x["gold"] for x, b in zip(ev, p_cl)]))
    eligible = [(x, s) for x, s, c in zip(ev, p_sc, p_cl) if c == x["gold"]]
    reliant = [x for x, s in eligible if s == x["label"]]
    rate = len(reliant) / max(1, len(eligible))
    pieces = len(tok(ev[0]["sc_word"], add_special_tokens=False)["input_ids"])
    rel = {"run_id": r["run_id"], "model": r["model"], "cond": r["cond"], "seed": r["seed"], "pieces": pieces,
           "reliance": rate, "n_eval": len(ev), "n_eligible": len(eligible), "n_reliant": len(reliant),
           "clean_test_acc": clean_acc, "val_shortcut_acc": val_sc_acc, "gate_pass": rate >= 0.90}
    json.dump(rel, open(os.path.join(out_dir, "reliance.json"), "w"), indent=1)
    print(f"   reliance {rate:.3f} ({len(reliant)}/{len(eligible)}) | clean test acc {clean_acc:.3f} | "
          f"pieces {pieces} | train {t_train/60:.1f} min")
    lime_exp = lime_explainer(2, r["seed"]); rng = np.random.default_rng(r["seed"])
    gt_rows, cp_rows = [], []
    t0 = time.time()
    for i, ex in enumerate(reliant[:N_ATTR]):
        ids, wid, words = st.encode(tok, ex["text"], args.max_len)
        if ex["sc_pos"] >= len(words) or words[ex["sc_pos"]] != ex["sc_word"]:
            continue
        gt_pos = {p for p, w in enumerate(wid) if w == ex["sc_pos"]}
        y = ex["label"]
        p_full = st.probs_batch(model, [st.emb(model, ids)[0]])[0]
        S, _ = st.attribute_all(model, tok, ids, wid, words, y, lime_exp, args, rng)
        base = {"run_id": r["run_id"], "model": r["model"], "cond": r["cond"], "seed": r["seed"], "ex": ex["id"]}
        for meth, sc in S.items():
            if sc is None:
                continue
            for gran, (us, groups) in st.unit_scores(meth, np.asarray(sc), wid, len(words)).items():
                m = st.gt_metrics(us, groups, gt_pos)
                if m:
                    gt_rows.append({**base, "pieces": len(gt_pos), "method": meth, "gran": gran, **m})
        cp_rows += comp_rows(model, tok, ids, wid, words, S, y, p_full, base)
        if i % 50 == 0:
            print(f"   ex {i}/{min(len(reliant), N_ATTR)}  {(time.time()-t0)/60:.1f} min")
    pd.DataFrame(gt_rows).to_csv(os.path.join(out_dir, "gt_metrics.csv.gz"), index=False)
    pd.DataFrame(cp_rows).to_csv(os.path.join(out_dir, "comp.csv.gz"), index=False)
    del model; torch.cuda.empty_cache() if st.DEV == "cuda" else None
    return {"train_min": t_train / 60, "attr_eval_min": (time.time() - t0) / 60, **rel}


def run_secondary(r, a, args, out_dir):
    from sklearn.metrics import f1_score
    d = os.path.join(a.splits, r["dataset"])
    lmap = json.load(open(os.path.join(d, "label_map.json")))
    tr, va, te = (st.read_jsonl(os.path.join(d, f"{s}.jsonl")) for s in ["train", "val", "test"])
    attr = random.Random(SEC_ATTR_SEED).sample(te, min(N_ATTR, len(te)))
    st.set_seed(r["seed"])
    tok, model = load_model(r["model"], len(lmap))
    t0 = time.time(); val_acc = st.finetune(model, tok, list(tr), va, args); t_train = time.time() - t0
    model.float().eval()
    pred = st.predict(model, tok, [x["text"] for x in te], args.max_len)
    gold = [x["label"] for x in te]
    perf = {"val_acc": val_acc, "test_acc": float(np.mean([p == g for p, g in zip(pred, gold)])),
            "test_macro_f1": float(f1_score(gold, pred, average="macro"))}
    json.dump(perf, open(os.path.join(out_dir, "performance.json"), "w"), indent=1)
    print(f"   test acc {perf['test_acc']:.3f} | macro-F1 {perf['test_macro_f1']:.3f} | train {t_train/60:.1f} min")
    lime_exp = lime_explainer(len(lmap), r["seed"]); rng = np.random.default_rng(r["seed"])
    rows, t0 = [], time.time()
    for i, ex in enumerate(attr):
        ids, wid, words = st.encode(tok, ex["text"], args.max_len)
        if not words:
            continue
        p_full = st.probs_batch(model, [st.emb(model, ids)[0]])[0]
        y = int(p_full.argmax())
        S, _ = st.attribute_all(model, tok, ids, wid, words, y, lime_exp, args, rng)
        n_sub = sum(1 for w in wid if w is not None)
        base = {"run_id": r["run_id"], "model": r["model"], "dataset": r["dataset"], "seed": r["seed"],
                "ex": ex["id"], "fertility": n_sub / len(words)}
        rows += comp_rows(model, tok, ids, wid, words, S, y, p_full, base)
        if i % 50 == 0:
            print(f"   ex {i}/{len(attr)}  {(time.time()-t0)/60:.1f} min")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "comp.csv.gz"), index=False)
    del model; torch.cuda.empty_cache() if st.DEV == "cuda" else None
    return {"train_min": t_train / 60, "attr_eval_min": (time.time() - t0) / 60, **perf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["primary", "secondary"])
    ap.add_argument("--shortcut_dir"); ap.add_argument("--splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="1/1", help="i/n: run only every n-th run starting at i")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    if a.group == "primary" and not a.shortcut_dir:
        sys.exit("--shortcut_dir required for primary")
    if a.group == "secondary" and not a.splits:
        sys.exit("--splits required for secondary")
    if a.group == "primary":
        for c in ["cross", "bb_high", "xlmr_low"]:
            if not os.path.exists(os.path.join(a.shortcut_dir, f"eval_{c}_test.jsonl")):
                sys.exit(f"missing eval_{c}_test.jsonl — build it with make_shortcut_data.py --eval_split test "
                         f"(Phase 4 only, after the prereg commit)")
    i, n = map(int, a.shard.split("/"))
    runs = run_list(a.group)[i - 1::n]
    args = types.SimpleNamespace(**CFG)
    root = os.path.join(a.out, a.group); os.makedirs(root, exist_ok=True)
    todo = [r for r in runs if not os.path.exists(os.path.join(root, r["run_id"], "DONE.json"))]
    print(f"device={st.DEV} | {a.group}: {len(runs)} runs in shard {a.shard}, {len(runs)-len(todo)} done, {len(todo)} to do")
    if a.dry_run:
        for r in todo:
            print("  ", r["run_id"])
        return
    for r in todo:
        out_dir = os.path.join(root, r["run_id"]); os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {r['run_id']}  ({datetime.now(BST):%Y-%m-%d %H:%M} BST)")
        t0 = time.time()
        info = run_primary(r, a, args, out_dir) if a.group == "primary" else run_secondary(r, a, args, out_dir)
        info.update({"run_id": r["run_id"], "finished_bst": f"{datetime.now(BST):%Y-%m-%d %H:%M}",
                     "gpu": torch.cuda.get_device_name(0) if st.DEV == "cuda" else "cpu",
                     "host": socket.gethostname(), "total_min": (time.time() - t0) / 60, "cfg": CFG})
        json.dump(info, open(os.path.join(out_dir, "DONE.json"), "w"), indent=1, default=str)  # checkpoint marker
        log = os.path.join(a.out, "compute_log.csv")
        pd.DataFrame([{"date_bst": info["finished_bst"], "run_id": r["run_id"], "group": a.group,
                       "gpu": info["gpu"], "gpu_hours": round(info["total_min"] / 60, 3)}]) \
            .to_csv(log, mode="a", header=not os.path.exists(log), index=False)
        print(f"   saved -> {out_dir}  ({info['total_min']:.1f} min)")
    print("\nAll runs in this shard are done. Next: python scripts/analyze_phase4.py --results <out dirs...>")


if __name__ == "__main__":
    main()
