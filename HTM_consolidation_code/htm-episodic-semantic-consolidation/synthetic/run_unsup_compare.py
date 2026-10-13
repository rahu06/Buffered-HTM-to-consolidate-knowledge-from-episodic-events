"""Revised mechanism vs baselines in the fully unsupervised setting (pretrained encoder, no contrastive projection),
synthetic benchmark (5 folds x 5 seeds) and CLINC150 (5 partitions x 5 seeds); same protocols as Tables 11 and 12."""
import sys, json, itertools, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.'); sys.path.insert(0, './synthetic')
import run_synthetic as RS
import run_clinc as R
import htm_v2
from htm_v2 import GloveEncoder, run_v2

SYN_G = dict(htm_v2._G)
CL_G = None


def syn_fold(f):
    htm_v2._G = SYN_G
    rows = []; v = (f + 1) % 5; S = RS.S
    tr = [s for s in S if s['fold'] not in (f, v) and s['target_fact']]
    va = [s for s in S if s['fold'] == v]; te = [s for s in S if s['fold'] == f]
    txt, lab, _ = RS.episodes(tr, 'train')
    enc = GloveEncoder().fit(txt); X = enc(txt)
    var0 = np.mean([X[lab == l].var(0) for l in set(lab)], axis=0)
    vf = [s['target_fact'] for s in va if s['target_fact']]; tf = [s['target_fact'] for s in te if s['target_fact']]
    val = [RS.stream(va, 'val', s) for s in RS.SEEDS[:3]]; cv = enc(vf); ct = enc(tf)
    best = RS.select(enc, var0, val, vf, dict(routing='similarity', merge=True), {'tau_merge': [0.75, 0.85, 0.92, 0.96]})
    for s in RS.SEEDS:
        t, y = RS.stream(te, 'test', s)
        rows.append(dict(dataset='synthetic', unit=f, seed=s, method='Revised mechanism (C)', **R.evaluate(y, *run_v2(enc(t), var0, s, **best)[:3], ct, tf)))
    for bn in RS.BASELINES:
        p = None
        if bn in RS.BGRID:
            p = max(RS.BGRID[bn], key=lambda q: np.mean([R.evaluate(y, *RS.baseline(bn, enc(t), len(vf), s, q), cv, vf)['BcF1'] for (t, y), s in zip(val, RS.SEEDS[:3])]))
        for s in RS.SEEDS:
            t, y = RS.stream(te, 'test', s)
            rows.append(dict(dataset='synthetic', unit=f, seed=s, method=bn, **R.evaluate(y, *RS.baseline(bn, enc(t), len(tf), s, p), ct, tf)))
    return rows


def clinc_part(part):
    g = np.load('./glove_clinc.npy', allow_pickle=True).item()
    htm_v2._G = dict(zip(g['words'], g['vecs']))
    rows = []
    rng = np.random.default_rng(1000 + part); perm = rng.permutation(R.INTENTS)
    tr_i, va_i, te_i = perm[:90], perm[90:120], perm[120:]
    tr = [t for i in tr_i for t in R.TR[i]]; lab = np.array([i for i in tr_i for _ in R.TR[i]])
    enc = GloveEncoder().fit(tr); X = enc(tr)
    var0 = np.mean([X[lab == i].var(0) for i in tr_i], axis=0)
    val = [R.make_stream(va_i, R.VA, R.OOS_VAL, s) for s in R.VAL_SEEDS]
    cv = enc([R.fact_text(i) for i in va_i]); ct = enc([R.fact_text(i) for i in te_i])
    grid = [dict(routing='similarity', merge=True, tau=t, tau_merge=m, z_thresh=np.inf) for t, m in itertools.product([0.15, 0.25, 0.35, 0.45, 0.55, 0.65], [0.75, 0.85, 0.92])]
    best = max(grid, key=lambda kw: np.mean([R.evaluate(y, *run_v2(enc(t), var0, s, **kw)[:3], cv, list(va_i))['BcF1'] for (t, y), s in zip(val, R.VAL_SEEDS)]))
    bgrid = {'BIRCH (online)': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'Agglomerative (cosine)': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'DBSCAN (cosine)': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]}
    names = {'K-Means (true number of facts)': 'K-Means (oracle k)', 'Streaming mini-batch K-Means (true number of facts)': 'Streaming mini-batch K-Means (oracle k)'}
    bp = {bn: max(gr, key=lambda q: np.mean([R.evaluate(y, *R.run_baseline(bn, enc(t), len(va_i), s, q), cv, list(va_i))['BcF1'] for (t, y), s in zip(val, R.VAL_SEEDS)])) for bn, gr in bgrid.items()}
    for s in R.TEST_SEEDS:
        t, y = R.make_stream(te_i, R.TE, R.OOS_TEST, s); Xt = enc(t)
        rows.append(dict(dataset='CLINC150', unit=part, seed=s, method='Revised mechanism (C)', **R.evaluate(y, *run_v2(Xt, var0, s, **best)[:3], ct, list(te_i))))
        for bn in RS.BASELINES:
            rows.append(dict(dataset='CLINC150', unit=part, seed=s, method=bn, **R.evaluate(y, *R.run_baseline(names.get(bn, bn), Xt, len(te_i), s, bp.get(bn)), ct, list(te_i))))
    return rows


def job(k):
    return syn_fold(k) if k < 5 else clinc_part(k - 5)


if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(job, range(10))
    json.dump([r for o in out for r in o], open('./synthetic/unsup_compare_results.json', 'w'), indent=1, default=float)
    print('done', f'{time.time()-t0:.0f}s')
