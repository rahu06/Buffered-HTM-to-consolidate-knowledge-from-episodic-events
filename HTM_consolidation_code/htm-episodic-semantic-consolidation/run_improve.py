"""Improved Stage 1: update rule and gates selectable, plus consolidation-time verification (kappa).
Selection on the validation fold/intents by B-cubed F1 (same rule as for every baseline); evaluation once on the test
streams (same seeds, encoder and READY rule as Tables 11/12 and the Comment 1.20 baselines)."""
import sys, json, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.'); sys.path.insert(0, './synthetic')
import run_clinc as R
import htm_v2
from htm_v2 import GloveEncoder, Stage1v2, verify
from htm_consolidation import train_contrastive

KAPPAS = [None, 0.3, 0.4, 0.5, 0.6, 0.7]


def stage1(X, var0, seed, kw):
    m = Stage1v2(X.shape[1], var0, seed=seed, **kw)
    raw = [m.step(x) for x in X]
    return m.finish(raw), m


def job(spec):
    ds, unit = spec; t0 = time.time(); rows = []
    if ds == 'synthetic':
        import run_synthetic as RS
        g = np.load('./synthetic/glove_synthetic.npy', allow_pickle=True).item(); htm_v2._G = dict(zip(g['words'], g['vecs']))
        S = RS.S; f = unit; v = (f + 1) % 5
        tr_s = [s for s in S if s['fold'] not in (f, v) and s['target_fact']]; va_s = [s for s in S if s['fold'] == v]; te_s = [s for s in S if s['fold'] == f]
        tr_txt, tr_lab, _ = RS.episodes(tr_s, 'train')
        va_facts = [s['target_fact'] for s in va_s if s['target_fact']]; te_facts = [s['target_fact'] for s in te_s if s['target_fact']]
        val = [RS.stream(va_s, 'val', s) for s in RS.SEEDS[:3]]; test = [RS.stream(te_s, 'test', s) for s in RS.SEEDS]
        enc = GloveEncoder().fit(tr_txt); enc.proj = train_contrastive(enc.base(tr_txt), list(tr_lab), steps=600, batch=min(128, len(set(tr_lab))), seed=f)
        base = dict(n_columns=256, n_ready=RS.N_READY); taus = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95]
        va_names, te_names, seeds = va_facts, te_facts, RS.SEEDS
    else:
        g = np.load('./glove_clinc.npy', allow_pickle=True).item(); htm_v2._G = dict(zip(g['words'], g['vecs']))
        rng = np.random.default_rng(1000 + unit); perm = rng.permutation(R.INTENTS); tr_i, va_i, te_i = perm[:90], perm[90:120], perm[120:]
        tr_txt = [t for i in tr_i for t in R.TR[i]]; tr_lab = np.array([i for i in tr_i for _ in R.TR[i]])
        va_facts = [R.fact_text(i) for i in va_i]; te_facts = [R.fact_text(i) for i in te_i]
        val = [R.make_stream(va_i, R.VA, R.OOS_VAL, s) for s in R.VAL_SEEDS]; test = [R.make_stream(te_i, R.TE, R.OOS_TEST, s) for s in R.TEST_SEEDS]
        enc = GloveEncoder().fit(tr_txt); enc.proj = train_contrastive(enc.base(tr_txt), list(tr_lab), seed=unit)
        base = dict(n_columns=1024, n_ready=3); taus = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        va_names, te_names, seeds = list(va_i), list(te_i), R.TEST_SEEDS
    Xtr = enc(tr_txt); var0 = np.mean([Xtr[tr_lab == l].var(0) for l in set(tr_lab)], axis=0)
    Xval = [(enc(t), y) for t, y in val]; Xte = [(enc(t), y) for t, y in test]; cva, cte = enc(va_facts), enc(te_facts)
    n_ready = base['n_ready']
    combos = []
    for tau, z, w, an, rs, mg in itertools.product(taus, [1.0, 2.0, np.inf], ['confidence', 'mean'], [True, False], [True, False], [None, 0.75, 0.85, 0.92]):
        combos.append(dict(base, routing='similarity', tau=tau, z_thresh=z, weighting=w, use_anchor=an, use_reservoir=rs,
                           merge=mg is not None, tau_merge=mg or 0.85))
    scores = {}                                     # (combo index, kappa) -> mean validation BcF1
    for ci, kw in enumerate(combos):
        acc = {k: [] for k in KAPPAS}
        for (X, y), s in zip(Xval, seeds[:3]):
            (a, r, T), _ = stage1(X, var0, s, kw)
            for k in KAPPAS:
                aa, rr, TT = (a, r, T) if k is None else verify(X, a, r, T, k, n_ready)
                acc[k].append(R.evaluate(y, aa, rr, TT, cva, va_names)['BcF1'])
        for k in KAPPAS:
            scores[(ci, k)] = np.mean(acc[k])
    print(ds, unit, 'selection done', f'{time.time()-t0:.0f}s', flush=True)
    variants = {'Improved mechanism (all changes)': lambda key: True,
                'Improved, without verification': lambda key: key[1] is None,
                'Improved, confidence-weighted updating only': lambda key: combos[key[0]]['weighting'] == 'confidence'}
    for vn, ok in variants.items():
        ci, k = max((key for key in scores if ok(key)), key=lambda key: scores[key])
        kw = combos[ci]
        for (X, y), s in zip(Xte, seeds):
            t1 = time.perf_counter(); (a, r, T), m = stage1(X, var0, s, kw)
            if k is not None: a, r, T = verify(X, a, r, T, k, n_ready)
            dt = time.perf_counter() - t1
            rows.append(dict(dataset=ds, unit=unit, seed=s, method=vn, kappa=k, runtime=dt, n_episodes=len(X),
                             config={kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv) for kk, vv in kw.items()},
                             **R.evaluate(y, a, r, T, cte, te_names)))
    print(ds, unit, 'done', f'{time.time()-t0:.0f}s', flush=True)
    return rows


if __name__ == '__main__':
    specs = [('synthetic', f) for f in range(5)] + [('CLINC150', p) for p in range(5)]
    with Pool(2) as pool:
        out = pool.map(job, specs)
    json.dump([r for o in out for r in o], open('./improve_results.json', 'w'), default=float)
    print('done')
