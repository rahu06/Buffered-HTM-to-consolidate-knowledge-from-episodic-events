"""Improved Stage 1 mechanism: component switches, update rule, thresholds and a compactness readiness test are all
selected on validation data (B-cubed F1, as for every other method), then evaluated once on the test streams.
Same protocols, encoders, streams and READY rules as Tables 11/12 and run_c120.py."""
import sys, json, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.'); sys.path.insert(0, './synthetic')
import run_clinc as R
import htm_v2
from htm_v2 import GloveEncoder, run_v2
from htm_consolidation import train_contrastive

READY_SIMS = [None, 0.5, 0.6, 0.7, 0.8]


def compact_ready(m, ready, rs):
    if rs is None or not len(ready):
        return ready
    keep = [j for j in ready if np.mean([r @ (m.mu[j] / (np.linalg.norm(m.mu[j]) + 1e-12)) for r in m.res[j]]) >= rs]
    return np.array(keep, dtype=int)


def evaluate_rs(y, a, ready, m, rs, cat, names):
    rdy = compact_ready(m, ready, rs)
    T = m.mu[rdy] / (np.linalg.norm(m.mu[rdy], axis=1, keepdims=True) + 1e-12) if len(rdy) else np.zeros((0, m.mu.shape[1]))
    return R.evaluate(y, a, rdy, T, cat, names)


def setup(ds, unit):
    if ds == 'synthetic':
        import run_synthetic as RS
        g = np.load('./synthetic/glove_synthetic.npy', allow_pickle=True).item(); htm_v2._G = dict(zip(g['words'], g['vecs']))
        S = RS.S; f = unit; v = (f + 1) % 5
        tr_s = [s for s in S if s['fold'] not in (f, v) and s['target_fact']]; va_s = [s for s in S if s['fold'] == v]; te_s = [s for s in S if s['fold'] == f]
        tr_txt, tr_lab, _ = RS.episodes(tr_s, 'train')
        va_f = [s['target_fact'] for s in va_s if s['target_fact']]; te_f = [s['target_fact'] for s in te_s if s['target_fact']]
        enc = GloveEncoder().fit(tr_txt); enc.proj = train_contrastive(enc.base(tr_txt), list(tr_lab), steps=600, batch=min(128, len(set(tr_lab))), seed=f)
        return dict(enc=enc, tr_txt=tr_txt, tr_lab=tr_lab, val=[RS.stream(va_s, 'val', s) for s in RS.SEEDS[:3]], test=[RS.stream(te_s, 'test', s) for s in RS.SEEDS],
                    va_f=va_f, te_f=te_f, va_n=va_f, te_n=te_f, seeds=RS.SEEDS, base=dict(n_columns=256, n_ready=RS.N_READY),
                    taus=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95])
    g = np.load('./glove_clinc.npy', allow_pickle=True).item(); htm_v2._G = dict(zip(g['words'], g['vecs']))
    rng = np.random.default_rng(1000 + unit); perm = rng.permutation(R.INTENTS); tr_i, va_i, te_i = perm[:90], perm[90:120], perm[120:]
    tr_txt = [t for i in tr_i for t in R.TR[i]]; tr_lab = np.array([i for i in tr_i for _ in R.TR[i]])
    enc = GloveEncoder().fit(tr_txt); enc.proj = train_contrastive(enc.base(tr_txt), list(tr_lab), seed=unit)
    return dict(enc=enc, tr_txt=tr_txt, tr_lab=tr_lab, val=[R.make_stream(va_i, R.VA, R.OOS_VAL, s) for s in R.VAL_SEEDS],
                test=[R.make_stream(te_i, R.TE, R.OOS_TEST, s) for s in R.TEST_SEEDS], va_f=[R.fact_text(i) for i in va_i], te_f=[R.fact_text(i) for i in te_i],
                va_n=list(va_i), te_n=list(te_i), seeds=R.TEST_SEEDS, base=dict(), taus=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65])


def job(spec):
    ds, unit = spec; t0 = time.time(); D = setup(ds, unit); enc = D['enc']
    Xtr = enc(D['tr_txt']); var0 = np.mean([Xtr[D['tr_lab'] == l].var(0) for l in set(D['tr_lab'])], axis=0)
    Xval = [(enc(t), y) for t, y in D['val']]; Xte = [(enc(t), y) for t, y in D['test']]; cva, cte = enc(D['va_f']), enc(D['te_f'])
    best, bs = None, -1
    for tau, w, an, rv, z, (mg, tm) in itertools.product(D['taus'], ['confidence', 'mean'], [True, False], [True, False], [np.inf, 1.0, 2.0],
                                                          [(False, 0.85), (True, 0.75), (True, 0.85), (True, 0.92)]):
        kw = dict(D['base'], routing='similarity', tau=tau, weighting=w, use_anchor=an, use_reservoir=rv, z_thresh=z, merge=mg, tau_merge=tm)
        runs = [(y, *run_v2(X, var0, s, **kw)) for (X, y), s in zip(Xval, D['seeds'][:3])]
        for rs in READY_SIMS:
            sc = np.mean([evaluate_rs(y, a, rdy, m, rs, cva, D['va_n'])['BcF1'] for y, a, rdy, T, m in runs])
            if sc > bs:
                bs, best = sc, (kw, rs)
    kw, rs = best; rows = []
    for (X, y), s in zip(Xte, D['seeds']):
        t1 = time.perf_counter(); a, rdy, T, m = run_v2(X, var0, s, **kw); dt = time.perf_counter() - t1
        rows.append(dict(dataset=ds, unit=unit, seed=s, method='Improved mechanism', runtime=dt, n_episodes=len(X),
                         selected={**{k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in kw.items()}, 'ready_sim': rs},
                         **evaluate_rs(y, a, rdy, m, rs, cte, D['te_n'])))
    print(ds, unit, f'{time.time()-t0:.0f}s', {k: kw[k] for k in ('tau', 'weighting', 'use_anchor', 'use_reservoir', 'z_thresh', 'merge')}, 'ready_sim', rs, flush=True)
    return rows


if __name__ == '__main__':
    specs = [('synthetic', f) for f in range(5)] + [('CLINC150', p) for p in range(5)]
    with Pool(2) as pool:
        out = pool.map(job, specs)
    json.dump([r for o in out for r in o], open('./improved_results.json', 'w'), default=float)
    print('done')
