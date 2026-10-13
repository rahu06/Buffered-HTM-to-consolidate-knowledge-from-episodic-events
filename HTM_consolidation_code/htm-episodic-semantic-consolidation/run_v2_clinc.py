"""Ablation of the Stage 1 improvements on CLINC150 (same intent-disjoint protocol as run_clinc.py)."""
import json, itertools, time, sys
import numpy as np
from multiprocessing import Pool
import run_clinc as R
from htm_consolidation import Encoder, train_contrastive
from htm_v2 import GloveEncoder, run_v2

TAUS, ZS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65], [2.0, np.inf]
ABL = {   # each step adds one change to the previous one
 'A. Original mechanism':            (dict(), {}),
 'B. + similarity routing':          (dict(routing='similarity'), {}),
 'C. + column merging':              (dict(routing='similarity', merge=True), {'tau_merge': [0.75, 0.85, 0.92]}),
 'D. + adaptive vigilance':          (dict(routing='similarity', merge=True, adaptive=True), {'tau_merge': [0.75, 0.85, 0.92], 'k_sd': [1.5, 2.5]}),
 'E. + shrinkage gate & retirement (full)': (dict(routing='similarity', merge=True, adaptive=True, shrink_n0=5, retire_after=200), {'tau_merge': [0.75, 0.85, 0.92], 'k_sd': [1.5, 2.5]}),
}
BASELINES = ['K-Means (oracle k)', 'Streaming mini-batch K-Means (oracle k)', 'BIRCH (online)', 'Agglomerative (cosine)', 'DBSCAN (cosine)']
BGRID = {'BIRCH (online)': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'Agglomerative (cosine)': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
         'DBSCAN (cosine)': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]}


def grid(extra):
    keys = list(extra)
    for tau, z, *vals in itertools.product(TAUS, ZS, *[extra[k] for k in keys]):
        yield dict(tau=tau, z_thresh=z, **dict(zip(keys, vals)))


def one_partition(part):
    t0 = time.time(); rows = []
    rng = np.random.default_rng(1000 + part); perm = rng.permutation(R.INTENTS)
    tr_i, va_i, te_i = perm[:90], perm[90:120], perm[120:]
    tr = [t for i in tr_i for t in R.TR[i]]; lab = np.array([i for i in tr_i for _ in R.TR[i]])
    val_streams = [R.make_stream(va_i, R.VA, R.OOS_VAL, s) for s in R.VAL_SEEDS]
    encoders = {}
    e = Encoder(seed=part).fit(tr); e.proj = train_contrastive(e.base(tr), list(lab), seed=part); encoders['TF-IDF+SVD + contrastive'] = e
    e = GloveEncoder().fit(tr); encoders['GloVe-SIF (pretrained)'] = e
    e = GloveEncoder().fit(tr); e.proj = train_contrastive(e.base(tr), list(lab), seed=part); encoders['GloVe-SIF + contrastive'] = e
    for en, enc in encoders.items():
        Xtr = enc(tr); var0 = np.mean([Xtr[lab == i].var(0) for i in tr_i], axis=0)
        Xval = [(enc(t), y) for t, y in val_streams]; cval = enc([R.fact_text(i) for i in va_i])
        configs = ABL if en == 'GloVe-SIF + contrastive' else {k: ABL[k] for k in ['A. Original mechanism', 'E. + shrinkage gate & retirement (full)']}
        chosen = {}
        for cn, (fixed, extra) in configs.items():
            best, bs = None, -1
            for g in grid(extra):
                kw = dict(fixed, **g)
                sc = np.mean([R.evaluate(y, *run_v2(X, var0, s, **kw)[:3], cval, list(va_i))['BcF1'] for (X, y), s in zip(Xval, R.VAL_SEEDS)])
                if sc > bs:
                    bs, best = sc, kw
            chosen[cn] = best
        bparam = {}
        if en == 'GloVe-SIF + contrastive':
            for bn, gr in BGRID.items():
                bs, bv = -1, None
                for g in gr:
                    sc = np.mean([R.evaluate(y, *R.run_baseline(bn, X, len(va_i), s, g), cval, list(va_i))['BcF1'] for (X, y), s in zip(Xval, R.VAL_SEEDS)])
                    if sc > bs:
                        bs, bv = sc, g
                bparam[bn] = bv
        cte = enc([R.fact_text(i) for i in te_i])
        for regime, frac in [('standard (12.5% negatives)', 0.125), ('noisy (40% negatives)', 0.40)]:
            R.OOS_FRAC = frac
            for s in R.TEST_SEEDS:
                txt, y = R.make_stream(te_i, R.TE, R.OOS_TEST, s); X = enc(txt)
                for cn, kw in chosen.items():
                    a, rdy, T, m = run_v2(X, var0, s, **kw)
                    rows.append(dict(partition=part, seed=s, regime=regime, encoder=en, method=cn, params={k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in kw.items()}, **R.evaluate(y, a, rdy, T, cte, list(te_i))))
                for bn in (BASELINES if bparam or en == 'GloVe-SIF + contrastive' else []):
                    lab_b, rdy, T = R.run_baseline(bn, X, len(te_i), s, bparam.get(bn))
                    rows.append(dict(partition=part, seed=s, regime=regime, encoder=en, method=bn, params={'param': bparam.get(bn)}, **R.evaluate(y, lab_b, rdy, T, cte, list(te_i))))
            R.OOS_FRAC = 0.125
        print(f'partition {part} {en} done {time.time()-t0:.0f}s', {k: v.get('tau') for k, v in chosen.items()}, flush=True)
    return rows


if __name__ == '__main__':
    with Pool(2) as pool:
        out = pool.map(one_partition, range(R.N_PARTITIONS))
    rows = [r for o in out for r in o]
    json.dump(rows, open('./v2_results.json', 'w'), indent=1, default=float)
    print('done', len(rows))
