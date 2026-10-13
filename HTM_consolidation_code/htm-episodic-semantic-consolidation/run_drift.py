"""Reviewer 2, Comment 2.4: robustness to embedding drift over the stream (CLINC150).
Drift model: a random orthogonal basis U splits the embedding space into planes; episode t of an n-episode stream is
rotated by angle theta_t = theta_max * t / (n - 1) in every plane, so cos(x, x') = cos(theta_t) exactly. theta_max is
set by the end-of-stream similarity c_end in {1.0, 0.95, 0.9, 0.8, 0.7}. The fact catalogue is embedded with the
end-of-stream encoder (rotated by theta_max), as it would be when facts are decoded after the stream.
Methods use the settings selected on validation data in the main experiments."""
import sys, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
import run_clinc as R, run_improved as RI, new_baselines as NB
from htm_v2 import run_v2

IMP = {r['unit']: r['selected'] for r in json.load(open('./improved_results.json')) if r['dataset'] == 'CLINC150'}
V2 = json.load(open('./v2_results.json'))
C120 = [r for r in json.load(open('./c120_results_final.json')) if r['part'] == 'baselines' and r['dataset'] == 'CLINC150']
LEVELS = [1.0, 0.95, 0.9, 0.8, 0.7]


def rotate(X, thetas, U):
    Z = X @ U; out = Z.copy(); D = Z.shape[1] // 2 * 2
    c, s = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
    a, b = Z[:, 0:D:2], Z[:, 1:D:2]
    out[:, 0:D:2] = c * a - s * b; out[:, 1:D:2] = s * a + c * b
    return out @ U.T


def job(part):
    D = RI.setup('CLINC150', part); enc = D['enc']
    Xtr = enc(D['tr_txt']); var0 = np.mean([Xtr[D['tr_lab'] == l].var(0) for l in set(D['tr_lab'])], axis=0)
    lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
    te_n = D['te_n']; cat0 = enc(D['te_f'])
    selF = dict(IMP[part]); rsF = selF.pop('ready_sim'); kwF = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in selF.items()}
    kwC = [r['params'] for r in V2 if r['partition'] == part and r['method'] == 'C. + column merging' and r['encoder'] == 'GloVe-SIF + contrastive'][0]
    kwC = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in kwC.items()}
    par = {r['method']: r.get('param') for r in C120 if r['unit'] == part}

    def mech(X, s, kw, rs):
        a, rdy, T, m = run_v2(X, var0, s, **kw); rdy = RI.compact_ready(m, rdy, rs)
        T = m.mu[rdy] / (np.linalg.norm(m.mu[rdy], axis=1, keepdims=True) + 1e-12) if len(rdy) else np.zeros((0, X.shape[1]))
        return a, rdy, T
    methods = {
        'Final configuration (F, running mean)': lambda X, s: mech(X, s, kwF, rsF),
        'Revised mechanism (C, confidence-weighted)': lambda X, s: mech(X, s, kwC, None),
        'BIRCH (online)': lambda X, s: R.run_baseline('BIRCH (online)', X, 30, s, par['BIRCH (online)']),
        'Fuzzy ART': lambda X, s: R.clusters_to_ready(X, NB.fuzzy_art(X, par['Fuzzy ART'], lo, hi)),
        'DenStream': lambda X, s: R.clusters_to_ready(X, NB.denstream(X, par['DenStream'])),
        'Online nearest-prototype memory': lambda X, s: R.clusters_to_ready(X, NB.leader_memory(X, par['Online nearest-prototype memory'])),
    }
    rows = []
    for s in R.TEST_SEEDS:
        t, y = D['test'][R.TEST_SEEDS.index(s)]; X0 = enc(t); n = len(X0)
        U, _ = np.linalg.qr(np.random.default_rng(7000 + 10 * part + s).standard_normal((X0.shape[1], X0.shape[1])))
        for c_end in LEVELS:
            th = np.arccos(c_end) * np.arange(n) / (n - 1)
            X = rotate(X0, th, U); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            cat = rotate(cat0, np.full(len(cat0), np.arccos(c_end)), U)
            for m, f in methods.items():
                a, rdy, T = f(X, s)
                ev = R.evaluate(y, a, rdy, T, cat, te_n)
                rows.append(dict(partition=part, seed=s, c_end=c_end, method=m, **{k: ev[k] for k in ('FactAccuracy', 'Purity', 'BcF1', 'NegAbsorbed', 'Fragmentation')}))
    print('partition', part, 'done', flush=True)
    return rows


if __name__ == '__main__':
    with Pool(2) as p:
        out = p.map(job, range(5))
    json.dump([r for o in out for r in o], open('./drift_results.json', 'w'), default=float)
    print('done')
