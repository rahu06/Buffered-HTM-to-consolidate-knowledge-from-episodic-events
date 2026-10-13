"""Reviewer 1, Comment 1.20.
(1) Additional online / streaming baselines (Fuzzy ART, DenStream, CluStream, online nearest-prototype memory) on both
    datasets, under the protocols of Tables 11 and 12 (same encoder, streams, READY rule, evaluation and validation-based
    selection of one hyperparameter by B-cubed F1), with runtime per stream for every method.
(2) Full factorial ablation of the revised Stage 1 mechanism (similarity routing always on): contrastive projection,
    anchor gate, reservoir gate, precision gate, confidence-weighted updating (vs running mean), column merging and
    threshold retuning (validation-selected thresholds vs the original fixed values tau = 0.75, z = 1.0,
    tau_merge = 0.85) -> 2^7 = 128 configurations, 5 test seeds per fold/partition."""
import sys, json, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.'); sys.path.insert(0, './synthetic')
import run_clinc as R
import htm_v2
from htm_v2 import GloveEncoder, run_v2
from htm_consolidation import train_contrastive
import new_baselines as NB

NEW = {'Fuzzy ART': [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9], 'DenStream': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
       'CluStream (true number of facts)': [None], 'Online nearest-prototype memory': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
FACTORS = ['projection', 'anchor', 'reservoir', 'precision', 'confidence', 'merging', 'retuning']


def new_baseline(name, X, k, p, lo, hi, seed):
    if name == 'Fuzzy ART': return NB.fuzzy_art(X, p, lo, hi)
    if name == 'DenStream': return NB.denstream(X, p)
    if name.startswith('CluStream'): return NB.clustream(X, k, seed)
    return NB.leader_memory(X, p)


def cfg_kw(f, base, taus):
    """Fixed switches and the selection grid for one factorial configuration."""
    fixed = dict(base, routing='similarity', use_anchor=bool(f['anchor']), use_reservoir=bool(f['reservoir']),
                 weighting='confidence' if f['confidence'] else 'mean', merge=bool(f['merging']))
    if not f['retuning']:
        return [dict(fixed, tau=0.75, z_thresh=1.0 if f['precision'] else np.inf, tau_merge=0.85)]
    zs = [1.0, 2.0] if f['precision'] else [np.inf]
    tms = [0.75, 0.85, 0.92] if f['merging'] else [0.85]
    return [dict(fixed, tau=t, z_thresh=z, tau_merge=m) for t, z, m in itertools.product(taus, zs, tms)]


def job(spec, baselines_only=False):
    ds, unit = spec; t0 = time.time(); rows = []
    if ds == 'synthetic':
        import run_synthetic as RS
        g = np.load('./synthetic/glove_synthetic.npy', allow_pickle=True).item(); htm_v2._G = dict(zip(g['words'], g['vecs']))
        S = RS.S; f = unit; v = (f + 1) % 5
        tr_s = [s for s in S if s['fold'] not in (f, v) and s['target_fact']]; va_s = [s for s in S if s['fold'] == v]; te_s = [s for s in S if s['fold'] == f]
        tr_txt, tr_lab, _ = RS.episodes(tr_s, 'train')
        va_facts = [s['target_fact'] for s in va_s if s['target_fact']]; te_facts = [s['target_fact'] for s in te_s if s['target_fact']]
        val = [RS.stream(va_s, 'val', s) for s in RS.SEEDS[:3]]; test = [RS.stream(te_s, 'test', s) for s in RS.SEEDS]
        base = dict(n_columns=256, n_ready=RS.N_READY); min_size = RS.N_READY
        taus = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95]
        def make(proj):
            enc = GloveEncoder().fit(tr_txt)
            if proj: enc.proj = train_contrastive(enc.base(tr_txt), list(tr_lab), steps=600, batch=min(128, len(set(tr_lab))), seed=f)
            return enc
        res = json.load(open('./synthetic/synthetic_results.json'))
        Ckw = [r['params'] for r in res if r['fold'] == f and r['setting'].startswith('C.') and r['catalogue'].startswith('closed')][0]
        old = {r['setting']: r['params']['param'] for r in res if r['fold'] == f and 'params' in r and 'param' in r['params']}
        bfun = lambda bn, X, k, s, p: RS.baseline(bn, X, k, s, p)
        BL = RS.BASELINES; seeds = RS.SEEDS
    else:
        g = np.load('./glove_clinc.npy', allow_pickle=True).item(); htm_v2._G = dict(zip(g['words'], g['vecs']))
        part = unit; rng = np.random.default_rng(1000 + part); perm = rng.permutation(R.INTENTS)
        tr_i, va_i, te_i = perm[:90], perm[90:120], perm[120:]
        tr_txt = [t for i in tr_i for t in R.TR[i]]; tr_lab = np.array([i for i in tr_i for _ in R.TR[i]])
        va_facts = [R.fact_text(i) for i in va_i]; te_facts = [R.fact_text(i) for i in te_i]
        val = [R.make_stream(va_i, R.VA, R.OOS_VAL, s) for s in R.VAL_SEEDS]; test = [R.make_stream(te_i, R.TE, R.OOS_TEST, s) for s in R.TEST_SEEDS]
        va_names, te_names = list(va_i), list(te_i)
        base = dict(); min_size = 3; taus = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        def make(proj):
            enc = GloveEncoder().fit(tr_txt)
            if proj: enc.proj = train_contrastive(enc.base(tr_txt), list(tr_lab), seed=part)
            return enc
        res = json.load(open('./v2_results.json'))
        sel = [r for r in res if r['partition'] == part and r['encoder'] == 'GloVe-SIF + contrastive' and r['regime'].startswith('standard')]
        Ckw = [r['params'] for r in sel if r['method'] == 'C. + column merging'][0]
        bmap = {'K-Means (true number of facts)': 'K-Means (oracle k)', 'Streaming mini-batch K-Means (true number of facts)': 'Streaming mini-batch K-Means (oracle k)'}
        old = {bn: [r['params']['param'] for r in sel if r['method'] == bmap.get(bn, bn)][0] for bn in ['K-Means (true number of facts)', 'Streaming mini-batch K-Means (true number of facts)', 'BIRCH (online)', 'Agglomerative (cosine)', 'DBSCAN (cosine)']}
        bfun = lambda bn, X, k, s, p: R.run_baseline(bmap.get(bn, bn), X, k, s, p)
        BL = list(old); seeds = R.TEST_SEEDS
    if ds == 'synthetic':
        va_names, te_names = va_facts, te_facts
    ev = lambda y, a, rdy, T, cat, names: R.evaluate(y, a, rdy, T, cat, names)
    Ckw = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in Ckw.items()}

    # ------------------------------------------------------------ (1) baselines with runtime
    enc = make(True); Xtr = enc(tr_txt); var0 = np.mean([Xtr[tr_lab == l].var(0) for l in set(tr_lab)], axis=0)
    lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
    Xval = [(enc(t), y) for t, y in val]; Xte = [(enc(t), y) for t, y in test]; cva, cte = enc(va_facts), enc(te_facts)
    for name, grid in NEW.items():
        p = max(grid, key=lambda q: np.mean([ev(y, *R.clusters_to_ready(X, new_baseline(name, X, len(va_facts), q, lo, hi, s), min_size=min_size), cva, va_names)['BcF1'] for (X, y), s in zip(Xval, seeds[:3])])) if len(grid) > 1 else grid[0]
        for (X, y), s in zip(Xte, seeds):
            t1 = time.perf_counter(); lab = new_baseline(name, X, len(te_facts), p, lo, hi, s); dt = time.perf_counter() - t1
            rows.append(dict(part='baselines', dataset=ds, unit=unit, seed=s, method=name, param=p, runtime=dt, n_units=int(len(set(lab.tolist()))), n_episodes=len(X), **ev(y, *R.clusters_to_ready(X, lab, min_size=min_size), cte, te_names)))
    for (X, y), s in zip(Xte, seeds):
        t1 = time.perf_counter(); a, rdy, T, m = run_v2(X, var0, s, **Ckw); dt = time.perf_counter() - t1
        rows.append(dict(part='baselines', dataset=ds, unit=unit, seed=s, method='Revised mechanism (C)', runtime=dt, n_units=int(m.used.sum()), n_episodes=len(X), **ev(y, a, rdy, T, cte, te_names)))
        for bn in BL:
            t1 = time.perf_counter(); out = bfun(bn, X, len(te_facts), s, old[bn]); dt = time.perf_counter() - t1
            rows.append(dict(part='baselines', dataset=ds, unit=unit, seed=s, method=bn, param=old[bn], runtime=dt, n_units=int(len(set(np.asarray(out[0]).tolist()))), n_episodes=len(X), **ev(y, *out, cte, te_names)))
    print(ds, unit, 'baselines done', f'{time.time()-t0:.0f}s', flush=True)
    if baselines_only:
        return rows

    # ------------------------------------------------------------ (2) factorial ablation
    for proj in (0, 1):
        enc = make(bool(proj)); Xtr = enc(tr_txt); var0 = np.mean([Xtr[tr_lab == l].var(0) for l in set(tr_lab)], axis=0)
        Xval = [(enc(t), y) for t, y in val]; Xte = [(enc(t), y) for t, y in test]; cva, cte = enc(va_facts), enc(te_facts)
        for bits in itertools.product((0, 1), repeat=6):
            fac = dict(zip(FACTORS[1:], bits)); fac['projection'] = proj
            cands = cfg_kw(fac, base, taus)
            if len(cands) > 1:
                kw = max(cands, key=lambda c: np.mean([ev(y, *run_v2(X, var0, s, **c)[:3], cva, va_names)['BcF1'] for (X, y), s in zip(Xval, seeds[:3])]))
            else:
                kw = cands[0]
            for (X, y), s in zip(Xte, seeds):
                a, rdy, T, m = run_v2(X, var0, s, **kw)
                rows.append(dict(part='factorial', dataset=ds, unit=unit, seed=s, tau=kw['tau'], **fac, **ev(y, a, rdy, T, cte, te_names)))
        print(ds, unit, 'factorial projection', proj, f'{time.time()-t0:.0f}s', flush=True)
    return rows


def job_clu(spec):
    global NEW
    NEW = {'CluStream (true number of facts)': [None]}
    rows = job(spec, baselines_only=True)
    return rows


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'clu':
    specs = [('synthetic', f) for f in range(5)] + [('CLINC150', p) for p in range(5)]
    with Pool(2) as pool:
        out = pool.map(job_clu, specs)
    json.dump([r for o in out for r in o if r['method'].startswith('CluStream')], open('./c120_clustream.json', 'w'), default=float)
    print('done clu')
elif __name__ == '__main__':
    specs = [('synthetic', f) for f in range(5)] + [('CLINC150', p) for p in range(5)]
    with Pool(2) as pool:
        out = pool.map(job, specs)
    json.dump([r for o in out for r in o], open('./c120_results.json', 'w'), default=float)
    print('done')
