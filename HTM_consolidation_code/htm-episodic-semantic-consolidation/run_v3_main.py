"""Final mechanism with replay consolidation (G): replay, splitting and support-scaled readiness are added to the
validation-selected switches of configuration F and selected, together with tau, on validation data by B-cubed F1
(the same criterion as for every other method). Evaluated (1) on the main test streams of both datasets and (2) on
the scaling test (30-120 facts, encoder without projection, settings selected on the validation intents)."""
import sys, json, itertools, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.'); sys.path.insert(0, './synthetic')
import run_clinc as R, run_improved as RI
from htm_v2 import GloveEncoder
from htm_v3 import run_v3

IMP = {(r['dataset'], r['unit']): r['selected'] for r in json.load(open('./improved_results.json'))}
EX = [dict()] + [dict(replay_every=100, split_tau=st, ready_frac=rf) for st in (None, 0.5, 0.6, 0.7) for rf in (0.0, 0.2, 0.3)] + [dict(ready_frac=rf) for rf in (0.2, 0.3)]


def select(Xval, var0, base, taus, cva, names, seeds):
    best, bs = None, -1
    for tau, ex in itertools.product(taus, EX):
        kw = dict(base, tau=tau)
        sc = np.mean([R.evaluate(y, *run_v3(X, var0, s, **kw, **ex)[:3], cva, names)['BcF1'] for (X, y), s in zip(Xval, seeds)])
        if sc > bs: bs, best = sc, (kw, ex)
    return best


def job(spec):
    ds, unit = spec; t0 = time.time(); D = RI.setup(ds, unit); enc = D['enc']; rows = []
    Xtr = enc(D['tr_txt']); var0 = np.mean([Xtr[D['tr_lab'] == l].var(0) for l in set(D['tr_lab'])], axis=0)
    sel = dict(IMP[(ds, unit)]); sel.pop('ready_sim'); base = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in sel.items()}
    if ds == 'CLINC150': base['n_columns'] = 2048
    Xval = [(enc(t), y) for t, y in D['val']]; cva, cte = enc(D['va_f']), enc(D['te_f'])
    kw, ex = select(Xval, var0, base, D['taus'], cva, D['va_n'], D['seeds'][:3])
    for (t, y), s in zip(D['test'], D['seeds']):
        X = enc(t); t1 = time.perf_counter(); a, rdy, T, m = run_v3(X, var0, s, **kw, **ex); dt = time.perf_counter() - t1
        rows.append(dict(part='main', dataset=ds, unit=unit, seed=s, runtime=dt / len(X) * 1000, used=int(m.used.sum()),
                         selected=dict(tau=kw['tau'], **{k: v for k, v in ex.items()}), **R.evaluate(y, a, rdy, T, cte, D['te_n'])))
    if ds == 'CLINC150':
        rng0 = np.random.default_rng(1000 + unit); perm = rng0.permutation(R.INTENTS); tr_i, va_i, te_i = perm[:90], perm[90:120], perm[120:]
        enc0 = GloveEncoder().fit(D['tr_txt']); X0 = enc0(D['tr_txt']); v0 = np.mean([X0[D['tr_lab'] == l].var(0) for l in set(D['tr_lab'])], axis=0)
        val0 = [(enc0(t), y) for t, y in [R.make_stream(va_i, R.VA, R.OOS_VAL, s) for s in R.VAL_SEEDS]]; c0 = enc0([R.fact_text(i) for i in va_i])
        kw0, ex0 = select(val0, v0, dict(base, n_columns=2048), [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], c0, list(va_i), R.VAL_SEEDS)
        pool = np.concatenate([tr_i, te_i])
        for s in R.TEST_SEEDS:
            order = np.random.default_rng(9000 + 10 * unit + s).permutation(pool)
            for k in (30, 60, 90, 120):
                facts = list(order[:k]); t, y = R.make_stream(facts, R.TE, R.OOS_TEST, s); X = enc0(t); cf = enc0([R.fact_text(i) for i in facts])
                t1 = time.perf_counter(); a, rdy, T, m = run_v3(X, v0, s, **kw0, **ex0); dt = time.perf_counter() - t1
                rows.append(dict(part='scaling', dataset=ds, unit=unit, seed=s, k=k, runtime=dt / len(X) * 1000, used=int(m.used.sum()),
                                 selected=dict(tau=kw0['tau'], **ex0), **R.evaluate(y, a, rdy, T, cf, facts)))
    print(ds, unit, f'{time.time()-t0:.0f}s', 'main', kw['tau'], ex, flush=True)
    return rows


if __name__ == '__main__':
    specs = [('CLINC150', p) for p in range(5)] + [('synthetic', f) for f in range(5)]
    with Pool(2) as p:
        out = p.map(job, specs)
    json.dump([r for o in out for r in o], open('./v3_results.json', 'w'), default=float)
    print('done')
