"""Reviewer 2, Comment 2.4: scalability beyond 256 columns (CLINC150).
(a) Column pool size n_columns in {128, 256, 512, 1024, 2048} on the standard test streams (30 facts, final
    configuration F with its validation-selected settings).
(b) Number of facts k in {30, 60, 90, 120}: facts drawn from the 120 non-validation intents, encoder without the
    contrastive projection (so no fact has been used for supervised training), thresholds re-selected on the
    validation intents for this encoder; F with 2048 columns, BIRCH and the nearest-prototype memory as references.
Reported: fact accuracy, purity, B-cubed F1, negatives absorbed, columns used, episodes dropped, runtime."""
import sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
import run_clinc as R, run_improved as RI, new_baselines as NB
from htm_v2 import run_v2, GloveEncoder

IMP = {r['unit']: r['selected'] for r in json.load(open('./improved_results.json')) if r['dataset'] == 'CLINC150'}
POOLS = [128, 256, 512, 1024, 2048]
KS = [30, 60, 90, 120]


def mech(X, var0, s, kw, rs):
    t0 = time.perf_counter(); a, rdy, T, m = run_v2(X, var0, s, **kw); dt = time.perf_counter() - t0
    rdy = RI.compact_ready(m, rdy, rs)
    T = m.mu[rdy] / (np.linalg.norm(m.mu[rdy], axis=1, keepdims=True) + 1e-12) if len(rdy) else np.zeros((0, X.shape[1]))
    return a, rdy, T, dt, int(m.used.sum()), int(m.dropped)


def job(part):
    rows = []
    D = RI.setup('CLINC150', part); enc = D['enc']
    Xtr = enc(D['tr_txt']); var0 = np.mean([Xtr[D['tr_lab'] == l].var(0) for l in set(D['tr_lab'])], axis=0)
    sel = dict(IMP[part]); rs = sel.pop('ready_sim'); kw = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in sel.items()}
    cat = enc(D['te_f'])
    for s, (t, y) in zip(R.TEST_SEEDS, D['test']):
        X = enc(t)
        for n_col in POOLS:
            a, rdy, T, dt, used, drop = mech(X, var0, s, dict(kw, n_columns=n_col), rs)
            ev = R.evaluate(y, a, rdy, T, cat, D['te_n'])
            rows.append(dict(part='pool', partition=part, seed=s, n_columns=n_col, k=30, method='F', runtime=dt / len(X) * 1000, used=used, dropped=drop,
                             **{m: ev[m] for m in ('FactAccuracy', 'Purity', 'BcF1', 'NegAbsorbed')}))
    # ---- (b) number of facts, encoder without projection
    rng0 = np.random.default_rng(1000 + part); perm = rng0.permutation(R.INTENTS); tr_i, va_i, te_i = perm[:90], perm[90:120], perm[120:]
    enc0 = GloveEncoder().fit(D['tr_txt'])
    X0 = enc0(D['tr_txt']); v0 = np.mean([X0[D['tr_lab'] == l].var(0) for l in set(D['tr_lab'])], axis=0)
    val = [(enc0(t), y) for t, y in [R.make_stream(va_i, R.VA, R.OOS_VAL, s) for s in R.VAL_SEEDS]]; cva = enc0([R.fact_text(i) for i in va_i])
    base = dict(kw, n_columns=2048)
    best = max([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], key=lambda tau: np.mean([R.evaluate(y, *mech(X, v0, s, dict(base, tau=tau), rs)[:3], cva, list(va_i))['BcF1'] for (X, y), s in zip(val, R.VAL_SEEDS)]))
    kwb = dict(base, tau=best)
    pb = max([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], key=lambda q: np.mean([R.evaluate(y, *R.run_baseline('BIRCH (online)', X, 30, s, q), cva, list(va_i))['BcF1'] for (X, y), s in zip(val, R.VAL_SEEDS)]))
    pp = max([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], key=lambda q: np.mean([R.evaluate(y, *R.clusters_to_ready(X, NB.leader_memory(X, q)), cva, list(va_i))['BcF1'] for (X, y), s in zip(val, R.VAL_SEEDS)]))
    pool = np.concatenate([tr_i, te_i])
    for s in R.TEST_SEEDS:
        rng = np.random.default_rng(9000 + 10 * part + s); order = rng.permutation(pool)
        for k in KS:
            facts = list(order[:k]); t, y = R.make_stream(facts, R.TE, R.OOS_TEST, s); X = enc0(t); cat0 = enc0([R.fact_text(i) for i in facts])
            a, rdy, T, dt, used, drop = mech(X, v0, s, kwb, rs); ev = R.evaluate(y, a, rdy, T, cat0, facts)
            rows.append(dict(part='facts', partition=part, seed=s, k=k, method='F', runtime=dt / len(X) * 1000, used=used, dropped=drop, n_eps=len(X),
                             **{m: ev[m] for m in ('FactAccuracy', 'Purity', 'BcF1', 'NegAbsorbed')}))
            for name, f in [('BIRCH (online)', lambda: R.run_baseline('BIRCH (online)', X, k, s, pb)),
                            ('Online nearest-prototype memory', lambda: R.clusters_to_ready(X, NB.leader_memory(X, pp)))]:
                t0 = time.perf_counter(); out = f(); dt = time.perf_counter() - t0; ev = R.evaluate(y, *out, cat0, facts)
                rows.append(dict(part='facts', partition=part, seed=s, k=k, method=name, runtime=dt / len(X) * 1000, n_eps=len(X),
                                 **{m: ev[m] for m in ('FactAccuracy', 'Purity', 'BcF1', 'NegAbsorbed')}))
    print('partition', part, 'done; tau', best, flush=True)
    return rows


if __name__ == '__main__':
    with Pool(2) as p:
        out = p.map(job, range(5))
    json.dump([r for o in out for r in o], open('./scaling_results.json', 'w'), default=float)
    print('done')
