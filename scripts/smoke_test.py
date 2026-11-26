#!/usr/bin/env python
"""
scripts/smoke_test.py  —  P1 end-to-end smoke test (owner: M3)

Pipeline per model (BanglaBERT, XLM-R):
  quick fine-tune (subset, smoke only) -> 200 attribution examples ->
  FAs: Random, Random_B (sanity), Attention, InputXGrad, IG, LIME  (DeepLift only with --with_deeplift) ->
  granularity: subword | word_sum | word_max ->
  operators: delete | mask | zero ->
  comprehensiveness at k in {10,20,50}% -> AOPC, normalized by Random.

THE NUMBERS FROM THIS SCRIPT ARE NOT RESULTS. The models are deliberately undertrained.
The script only checks that the pipeline runs, checks the sanity criteria, and measures T4 runtime.

Requires the output of scripts/audit_data.py (data/splits/<dataset>/*.jsonl).
  pip install transformers==4.46.3 captum==0.7.0 lime==0.2.0.1 pandas
  python scripts/smoke_test.py --dataset bdshs --n_attr 200        # --split val (default)
"""
import argparse, json, math, os, random, time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = {"banglabert": "csebuetnlp/banglabert", "xlmr": "xlm-roberta-base"}
KS = [0.10, 0.20, 0.50]
GRANS = ["subword", "word_sum", "word_max"]
OPS = ["delete", "mask", "zero"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


# ------------------------------------------------------------------ encoding
def encode(tok, text, max_len):
    """Words = whitespace split (the same definition is used by LIME). Returns ids + word_ids."""
    words = text.split()
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=max_len)
    wid = enc.word_ids()
    n_words = max([w for w in wid if w is not None], default=-1) + 1  # words that survived truncation
    return enc["input_ids"], wid, words[:n_words]


# ------------------------------------------------------------------ training (smoke only)
def finetune(model, tok, train, val, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
    model.train()
    for ep in range(args.epochs):
        random.shuffle(train)
        for i in range(0, len(train), args.bs):
            b = train[i:i + args.bs]
            enc = tok([x["text"] for x in b], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(DEV)
            y = torch.tensor([x["label"] for x in b], device=DEV)
            with torch.autocast(device_type="cuda", enabled=DEV == "cuda"):
                loss = F.cross_entropy(model(**enc).logits, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(val), 64):
            b = val[i:i + 64]
            enc = tok([x["text"] for x in b], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(DEV)
            correct += (model(**enc).logits.argmax(-1).cpu() == torch.tensor([x["label"] for x in b])).sum().item()
    return correct / len(val)


# ------------------------------------------------------------------ forward via inputs_embeds (used everywhere)
class EmbedWrapper(nn.Module):
    """Every forward pass, including the clean one, goes through inputs_embeds, so all scores share one code path."""
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, embeds, mask):
        return self.model(inputs_embeds=embeds, attention_mask=mask).logits  # attribute the target LOGIT


@torch.no_grad()
def emb(model, ids):
    return model.get_input_embeddings()(torch.tensor([ids], device=DEV))


@torch.no_grad()
def probs_batch(model, embeds_list):
    """embeds_list: list of (L_i, D) tensors -> softmax probs (N, C). Right-padded with zeros + mask 0."""
    out = []
    for i in range(0, len(embeds_list), 64):
        chunk = embeds_list[i:i + 64]
        L = max(e.shape[0] for e in chunk); D = chunk[0].shape[1]
        E = torch.zeros(len(chunk), L, D, device=DEV); M = torch.zeros(len(chunk), L, dtype=torch.long, device=DEV)
        for j, e in enumerate(chunk):
            E[j, :e.shape[0]] = e; M[j, :e.shape[0]] = 1
        out.append(F.softmax(model(inputs_embeds=E, attention_mask=M).logits.float(), -1))
    return torch.cat(out)


# ------------------------------------------------------------------ attribution methods -> per-subword scores
def attribute_all(model, tok, ids, wid, words, target, lime_explainer, args, rng):
    from captum.attr import IntegratedGradients, InputXGradient, DeepLift
    wrap = EmbedWrapper(model).eval()
    x = emb(model, ids).detach().requires_grad_(True)
    m = torch.ones(1, len(ids), dtype=torch.long, device=DEV)
    special = torch.tensor([w is None for w in wid], device=DEV)
    base_ids = [i if s else tok.pad_token_id for i, s in zip(ids, special.tolist())]  # PAD baseline, specials kept
    base = emb(model, base_ids).detach()
    S, t0 = {}, {}

    def timed(name, fn):
        t = time.time()
        try:
            S[name] = fn()
        except Exception as e:  # log the failure; never silently drop a method
            print(f"   [FAIL] {name}: {type(e).__name__}: {str(e)[:160]}")
            S[name] = None
        t0[name] = time.time() - t

    L = len(ids)
    timed("Random", lambda: rng.random(L))
    timed("Random_B", lambda: rng.random(L))

    def attn():
        with torch.no_grad():
            a = model(inputs_embeds=x, attention_mask=m, output_attentions=True).attentions[-1]
        return a[0].mean(0)[0].float().cpu().numpy()  # last layer, mean heads, [CLS]/<s> row
    timed("Attention", attn)
    timed("InputXGrad", lambda: InputXGradient(wrap).attribute(x, target=target, additional_forward_args=(m,))
          .sum(-1)[0].detach().cpu().numpy())
    timed("IG", lambda: IntegratedGradients(wrap).attribute(
        x, baselines=base, target=target, additional_forward_args=(m,), n_steps=50,
        internal_batch_size=args.ig_batch).sum(-1)[0].detach().cpu().numpy())
    if args.with_deeplift:  # excluded from the main study (DEVIATIONS.md, 2026-11-27); kept for the appendix check
        timed("DeepLift", lambda: DeepLift(wrap).attribute(
            x, baselines=base, target=target, additional_forward_args=(m,)).sum(-1)[0].detach().cpu().numpy())

    def lime():
        text = " ".join(words)

        @torch.no_grad()
        def clf(strings):
            embs = []
            for s in strings:
                ii = tok(s.split(), is_split_into_words=True, truncation=True, max_length=args.max_len)["input_ids"]
                embs.append(emb(model, ii)[0])
            return probs_batch(model, embs).cpu().numpy()
        exp = lime_explainer.explain_instance(text, clf, labels=(target,), num_features=len(words),
                                              num_samples=args.lime_samples)
        wscore = np.zeros(len(words))
        for idx, w in exp.as_map()[target]:  # bow=False: idx = word position
            wscore[idx] = w
        return wscore  # NATIVE WORD-LEVEL (projected to subwords below)
    timed("LIME", lime)
    return S, t0


# ------------------------------------------------------------------ granularity + perturbation
def unit_scores(method, scores, wid, n_words):
    """Returns {granularity: (unit_scores, unit -> list of subword positions)}.
    Words that produced no subword tokens (e.g. a lone ZWNJ/ZWJ, or text lost to truncation)
    are dropped, so every unit has at least one position."""
    content = [p for p, w in enumerate(wid) if w is not None]
    by_word = defaultdict(list)
    for p in content:
        by_word[wid[p]].append(p)
    kept = [j for j in range(n_words) if by_word.get(j)]  # non-empty words only
    groups = [by_word[j] for j in kept]
    out = {}
    if not content or not kept:
        return out
    if method == "LIME":  # scores are word-level, indexed by whitespace word position
        out["subword"] = (np.array([scores[wid[p]] / len(by_word[wid[p]]) for p in content]), [[p] for p in content])
        w = np.array([scores[j] for j in kept])
        out["word_sum"] = out["word_max"] = (w, groups)
        return out
    out["subword"] = (np.array([scores[p] for p in content]), [[p] for p in content])
    if method.startswith("Random"):  # uniform random ranking at EVERY granularity: one draw per word
        w = np.array([scores[g[0]] for g in groups])  # (sum/max of uniforms would favour long words)
        out["word_sum"] = out["word_max"] = (w, groups)
        return out
    out["word_sum"] = (np.array([scores[g].sum() for g in groups]), groups)
    out["word_max"] = (np.array([scores[g].max() for g in groups]), groups)
    return out


def perturb(model, tok, ids, drop_pos, op):
    drop = set(drop_pos)
    if op == "delete":
        return emb(model, [t for p, t in enumerate(ids) if p not in drop])[0]
    if op == "mask":
        return emb(model, [tok.mask_token_id if p in drop else t for p, t in enumerate(ids)])[0]
    e = emb(model, ids)[0].clone()  # zero
    e[list(drop)] = 0.0
    return e


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bdshs", choices=["bdshs", "massive_bn"])
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--n_attr", type=int, default=200)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="split for attribution examples; keep 'val' until Phase 4 (test-set firewall)")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--ig_batch", type=int, default=10)
    ap.add_argument("--lime_samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with_deeplift", action="store_true",
                    help="re-enable DeepLift (excluded: Captum rules not applied to HF modules, ~= InputXGrad)")
    ap.add_argument("--out", default="results/smoke")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from lime.lime_text import LimeTextExplainer

    d = os.path.join(args.splits, args.dataset)
    lmap = json.load(open(os.path.join(d, "label_map.json")))
    train, val = (read_jsonl(os.path.join(d, f"{s}.jsonl")) for s in ["train", "val"])
    test = read_jsonl(os.path.join(d, "test.jsonl")) if args.split == "test" else None
    set_seed(args.seed)
    train = random.sample(train, min(args.n_train, len(train)))
    if args.split == "test":
        print("[WARN] using TEST split: allowed only in Phase 4, after the prereg commit")
    pool = val if args.split == "val" else test
    attr_set = random.Random(args.seed).sample(pool, min(args.n_attr, len(pool)))
    ft_val = [x for x in val if x["id"] not in {a["id"] for a in attr_set}][:1000]  # no overlap with attr set
    print(f"device={DEV} | {args.dataset}: {len(lmap)} classes | train subset {len(train)} | "
          f"attr {len(attr_set)} from {args.split} | fine-tune val {len(ft_val)}")

    rows, timing = [], []
    for short, hub in MODELS.items():
        set_seed(args.seed)
        print(f"\n=== {short} ({hub})")
        tok = AutoTokenizer.from_pretrained(hub, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            hub, num_labels=len(lmap), attn_implementation="eager").to(DEV)
        t = time.time(); acc = finetune(model, tok, list(train), ft_val, args)
        t_train = time.time() - t
        print(f"   smoke fine-tune: {t_train/60:.1f} min | val acc {acc:.3f} (undertrained on purpose)")
        model.float().eval()
        lime_exp = LimeTextExplainer(class_names=list(lmap), bow=False, split_expression=r"\s+",
                                     mask_string="", random_state=args.seed)
        # NOTE: split_expression must be whitespace. LIME's default r'\W+' splits Bangla words at
        # vowel signs (combining marks are not \w), which would destroy the word unit.
        rng = np.random.default_rng(args.seed)
        t_attr = t_eval = 0.0; mtime = defaultdict(float)
        for ex_i, ex in enumerate(attr_set):
            ids, wid, words = encode(tok, ex["text"], args.max_len)
            if not words:
                continue
            p_full = probs_batch(model, [emb(model, ids)[0]])[0]
            y = int(p_full.argmax())
            t = time.time()
            S, mt = attribute_all(model, tok, ids, wid, words, y, lime_exp, args, rng)
            t_attr += time.time() - t
            for k_, v in mt.items():
                mtime[k_] += v
            t = time.time()
            variants, keys = [], []  # emb() is no_grad, so variants hold no autograd graph
            for meth, sc in S.items():
                if sc is None:
                    continue
                for gran, (us, groups) in unit_scores(meth, np.asarray(sc), wid, len(words)).items():
                    order = np.argsort(-us, kind="stable")
                    for k in KS:
                        n_units = max(1, math.ceil(k * len(us)))
                        drop = [p for u in order[:n_units] for p in groups[u]]
                        for op in OPS:
                            variants.append(perturb(model, tok, ids, drop, op))
                            keys.append((meth, gran, op, k))
            P = probs_batch(model, variants)[:, y]
            for key, pk in zip(keys, P.tolist()):
                rows.append({"model": short, "ex": ex["id"], "method": key[0], "gran": key[1], "op": key[2],
                             "k": key[3], "comp": float(p_full[y]) - pk})
            t_eval += time.time() - t
            if ex_i % 50 == 0:
                print(f"   ex {ex_i}/{len(attr_set)}  attr {t_attr:.0f}s  eval {t_eval:.0f}s")
        n = len(attr_set)
        timing.append({"model": short, "train_min": t_train / 60, "attr_s_per_ex": t_attr / n,
                       "eval_s_per_ex": t_eval / n, **{f"{k}_s_per_ex": v / n for k, v in mtime.items()}})
        del model; torch.cuda.empty_cache()

    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.out, "comp_per_example.csv"), index=False)
    aopc = df.groupby(["model", "method", "gran", "op", "ex"])["comp"].mean().reset_index()  # AOPC over k
    raw = aopc.groupby(["model", "method", "gran", "op"])["comp"].mean().unstack("method")
    raw.to_csv(os.path.join(args.out, "aopc_comp_raw.csv"))
    delta = raw.sub(raw["Random"], axis=0)  # FA − Random: stable even when Random ≈ 0
    ratio = raw.div(raw["Random"], axis=0)  # Zhao & Aletras style; unstable if Random AOPC ≈ 0
    delta.to_csv(os.path.join(args.out, "aopc_comp_minus_random.csv"))
    ratio.to_csv(os.path.join(args.out, "aopc_comp_ratio_random.csv"))
    print("\nAOPC-comp RAW (SMOKE, NOT RESULTS):\n", raw.round(3).to_string())
    print("\nAOPC-comp minus Random:\n", delta.round(3).to_string())
    if (raw["Random"].abs() < 0.01).any():
        print("[WARN] Random AOPC ≈ 0 in some cells, so the ratio is unstable there. Report delta; "
              "log the choice in DEVIATIONS.md.")

    # ---------------- sanity checks (prereg §8 / Gate 3)
    # Two independent random rankings must be indistinguishable: 95% bootstrap CI of the paired
    # per-example difference (Random_B − Random) must include 0 in every cell.
    piv = aopc.pivot_table(index=["model", "gran", "op", "ex"], columns="method", values="comp")
    brng = np.random.default_rng(123); bad = []
    for cell, g in piv.groupby(level=[0, 1, 2]):
        d_ = (g["Random_B"] - g["Random"]).to_numpy()
        bs = [brng.choice(d_, len(d_)).mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        if not (lo <= 0 <= hi):
            bad.append(cell)
    failed = [m for m in ["Attention", "InputXGrad", "IG", "LIME"] + (["DeepLift"] if args.with_deeplift else [])
              if m not in raw.columns or raw[m].isna().any()]
    print(f"\n[SANITY] Random_B vs Random indistinguishable in all cells: {not bad} {bad if bad else ''}")
    print(f"[SANITY] methods with failures/missing cells: {failed or 'none'}")

    tm = pd.DataFrame(timing); tm.to_csv(os.path.join(args.out, "timing.csv"), index=False)
    print("\nTiming (seconds per example):\n", tm.round(2).to_string(index=False))
    per_ex = (tm["attr_s_per_ex"] + tm["eval_s_per_ex"]).sum()  # both models
    proj_h = per_ex * 500 * 5 * 2 / 3600  # 500 ex × 5 seeds × 2 datasets
    print(f"\n[ESTIMATE] attribution+eval for the full plan ≈ {proj_h:.1f} GPU-h "
          f"(+ fine-tuning: scale train_min by full-train size × epochs × 20 runs). Log in compute_log.csv.")


if __name__ == "__main__":
    main()
