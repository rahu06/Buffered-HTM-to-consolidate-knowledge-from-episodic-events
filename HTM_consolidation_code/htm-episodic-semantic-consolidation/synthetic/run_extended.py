"""Reviewer 1, Comment 1.10: extended metrics, 20 seeds per fold/partition, and order vs initialization sensitivity.
Thresholds are those selected on validation data in the main experiments (synthetic_results.json, v2_results.json),
so no test data are used for selection. Seeds 1..20: each seed sets the presentation order (and, for CLINC150, the
sample of out-of-scope utterances) and the random-projection initialization; the sensitivity runs vary one of the two
while holding the other fixed."""
import sys, json, time
import numpy as np
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, '.'); sys.path.insert(0, './synthetic')
import run_synthetic as RS
import run_clinc as R
import htm_v2
from htm_v2 import GloveEncoder, run_v2
from htm_consolidation import train_contrastive

SEEDS = list(range(1, 21))
SYN_G = dict(htm_v2._G)


def fix(kw):
    return {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in kw.items()}


def ext(y, assign, ready, T, cat, names):
    out = R.evaluate(y, assign, ready, T, cat, names)
    y = np.asarray(y); a = np.asarray(assign).copy()
    nxt = a.max() + 1 if a.size else 0
    for k in np.flatnonzero(a < 0):
        a[k] = nxt; nxt += 1
    pos = y != 'oos'; facts = set(y[pos])
    maj = {c: Counter(y[a == c]).most_common(1)[0][0] for c in ready}
    pos_cols = [c for c in ready if maj[c] != 'oos']
    covered = {maj[c] for c in pos_cols}
    out['MissedFactRate'] = 1 - len(covered) / len(facts)
    out['DuplicateRate'] = (len(pos_cols) - len(covered)) / max(len(pos_cols), 1)
    in_ready = np.isin(a, list(ready))
    out['EpisodeCoverage'] = float(in_ready[pos].mean())
    out['EpisodeCorrect'] = float(np.mean([in_ready[i] and maj.get(a[i]) == y[i] for i in np.flatnonzero(pos)]))
    out['n_facts'] = len(facts); out['n_pos_episodes'] = int(pos.sum()); out['n_neg_episodes'] = int((~pos).sum())
    return out


def syn_fold(f):
    htm_v2._G = SYN_G
    res = json.load(open('./synthetic/synthetic_results.json'))
    par = {r['setting']: r['params'] for r in res if r['fold'] == f and r['catalogue'].startswith('closed') and 'params' in r}
    rows = []; v = (f + 1) % 5; S = RS.S
    tr = [s for s in S if s['fold'] not in (f, v) and s['target_fact']]
    te = [s for s in S if s['fold'] == f]
    txt, lab, _ = RS.episodes(tr, 'train')
    enc = GloveEncoder().fit(txt)
    enc.proj = train_contrastive(enc.base(txt), list(lab), steps=600, batch=min(128, len(set(lab))), seed=f)
    X = enc(txt); var0 = np.mean([X[lab == l].var(0) for l in set(lab)], axis=0)
    tf = [s['target_fact'] for s in te if s['target_fact']]; ct = enc(tf)
    mech = {'Revised mechanism (C)': fix(par['C. + column merging (revised mechanism)']), 'Base configuration (A)': fix(par['A. Base configuration (random shortlist)'])}
    for s in SEEDS:
        t, y = RS.stream(te, 'test', s); Xt = enc(t)
        for m, kw in mech.items():
            rows.append(dict(dataset='synthetic', unit=f, seed=s, kind='main', method=m, **ext(y, *run_v2(Xt, var0, s, **kw)[:3], ct, tf)))
        for bn in RS.BASELINES:
            rows.append(dict(dataset='synthetic', unit=f, seed=s, kind='main', method=bn, **ext(y, *RS.baseline(bn, Xt, len(tf), s, par[bn]['param']), ct, tf)))
        for m, kw in mech.items():   # sensitivity: order varies (init fixed), init varies (order fixed)
            rows.append(dict(dataset='synthetic', unit=f, seed=s, kind='order', method=m, **ext(y, *run_v2(Xt, var0, 0, **kw)[:3], ct, tf)))
            t1, y1 = RS.stream(te, 'test', 1)
            rows.append(dict(dataset='synthetic', unit=f, seed=s, kind='init', method=m, **ext(y1, *run_v2(enc(t1), var0, s, **kw)[:3], ct, tf)))
    return rows


def clinc_part(part):
    g = np.load('./glove_clinc.npy', allow_pickle=True).item()
    htm_v2._G = dict(zip(g['words'], g['vecs']))
    res = json.load(open('./v2_results.json'))
    par = {r['method']: r['params'] for r in res if r['partition'] == part and r['encoder'] == 'GloVe-SIF + contrastive' and r['regime'].startswith('standard')}
    rows = []
    rng = np.random.default_rng(1000 + part); perm = rng.permutation(R.INTENTS)
    tr_i, te_i = perm[:90], perm[120:]
    tr = [t for i in tr_i for t in R.TR[i]]; lab = np.array([i for i in tr_i for _ in R.TR[i]])
    enc = GloveEncoder().fit(tr); enc.proj = train_contrastive(enc.base(tr), list(lab), seed=part)
    X = enc(tr); var0 = np.mean([X[lab == i].var(0) for i in tr_i], axis=0)
    ct = enc([R.fact_text(i) for i in te_i]); names = list(te_i)
    mech = {'Revised mechanism (C)': fix(par['C. + column merging']), 'Base configuration (A)': fix(par['A. Original mechanism'])}
    bmap = {'K-Means (true number of facts)': 'K-Means (oracle k)', 'Streaming mini-batch K-Means (true number of facts)': 'Streaming mini-batch K-Means (oracle k)'}
    for s in SEEDS:
        t, y = R.make_stream(te_i, R.TE, R.OOS_TEST, s); Xt = enc(t)
        for m, kw in mech.items():
            rows.append(dict(dataset='CLINC150', unit=part, seed=s, kind='main', method=m, **ext(y, *run_v2(Xt, var0, s, **kw)[:3], ct, names)))
        for bn in RS.BASELINES:
            nm = bmap.get(bn, bn)
            rows.append(dict(dataset='CLINC150', unit=part, seed=s, kind='main', method=bn, **ext(y, *R.run_baseline(nm, Xt, len(te_i), s, par[nm]['param']), ct, names)))
        for m, kw in mech.items():
            rows.append(dict(dataset='CLINC150', unit=part, seed=s, kind='order', method=m, **ext(y, *run_v2(Xt, var0, 0, **kw)[:3], ct, names)))
            t1, y1 = R.make_stream(te_i, R.TE, R.OOS_TEST, 1)
            rows.append(dict(dataset='CLINC150', unit=part, seed=s, kind='init', method=m, **ext(y1, *run_v2(enc(t1), var0, s, **kw)[:3], ct, names)))
    return rows


def job(k):
    return syn_fold(k) if k < 5 else clinc_part(k - 5)


if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(job, range(10))
    json.dump([r for o in out for r in o], open('./synthetic/extended_results.json', 'w'), default=float)
    print('done', f'{time.time()-t0:.0f}s')
