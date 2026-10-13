"""Reviewer 2, Comment 2.3: catastrophic forgetting on a non-stationary stream (CLINC150).
The 30 test intents of each partition are split into 3 phases of 10 intents; the stream presents phase 1, then phase 2,
then phase 3 (out-of-scope negatives, 12.5%, spread over all phases), and earlier intents never reappear. The same
episodes in fully shuffled order form the stationary control. At the end of the stream we measure fact accuracy on
the phase-1 facts (retention of old knowledge), the phase-3 facts (recent knowledge) and all facts.
Forgetting = phase-1 accuracy in the shuffled control minus phase-1 accuracy in the phased stream.
Methods use the settings selected on validation data in the main experiments (Tables 12, A14)."""
import sys, json
import numpy as np
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, '.')
import run_clinc as R, run_improved as RI, new_baselines as NB
from htm_v2 import run_v2

IMP = {r['unit']: r['selected'] for r in json.load(open('./improved_results.json')) if r['dataset'] == 'CLINC150'}
V2 = json.load(open('./v2_results.json'))
C120 = [r for r in json.load(open('./c120_results_final.json')) if r['part'] == 'baselines' and r['dataset'] == 'CLINC150']


def correct_facts(y, assign, ready, T, cat, names):
    y = np.asarray(y); a = np.asarray(assign).copy(); nxt = a.max() + 1
    for k in np.flatnonzero(a < 0): a[k] = nxt; nxt += 1
    if not len(ready): return set(), set()
    dec = [names[i] for i in np.argmax(T @ cat.T, axis=1)]
    corr, cov = set(), set()
    for c, d in zip(ready, dec):
        lab = Counter(y[a == c]).most_common(1)[0][0]
        if lab != 'oos':
            cov.add(lab)
            if d == lab: corr.add(lab)
    return corr, cov


def job(part):
    D = RI.setup('CLINC150', part); enc = D['enc']
    Xtr = enc(D['tr_txt']); var0 = np.mean([Xtr[D['tr_lab'] == l].var(0) for l in set(D['tr_lab'])], axis=0)
    lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
    te_i = list(D['te_n']); cat = enc(D['te_f'])
    selF = dict(IMP[part]); rsF = selF.pop('ready_sim'); kwF = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in selF.items()}
    kwC = [r['params'] for r in V2 if r['partition'] == part and r['method'] == 'C. + column merging' and r['encoder'] == 'GloVe-SIF + contrastive'][0]
    kwC = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in kwC.items()}
    par = {r['method']: r.get('param') for r in C120 if r['unit'] == part}
    methods = {
        'Final configuration (F)': lambda X, s: (lambda o: (o[0], RI.compact_ready(o[3], o[1], rsF), o[3]))(run_v2(X, var0, s, **kwF)),
        'Revised mechanism (C)': lambda X, s: (lambda o: (o[0], o[1], o[3]))(run_v2(X, var0, s, **kwC)),
        'BIRCH (online)': lambda X, s: R.run_baseline('BIRCH (online)', X, 30, s, par['BIRCH (online)']),
        'Streaming mini-batch K-Means (true number of facts)': lambda X, s: R.run_baseline('Streaming mini-batch K-Means (oracle k)', X, 30, s),
        'Fuzzy ART': lambda X, s: R.clusters_to_ready(X, NB.fuzzy_art(X, par['Fuzzy ART'], lo, hi)),
        'DenStream': lambda X, s: R.clusters_to_ready(X, NB.denstream(X, par['DenStream'])),
        'Online nearest-prototype memory': lambda X, s: R.clusters_to_ready(X, NB.leader_memory(X, par['Online nearest-prototype memory'])),
    }
    rows = []
    for s in R.TEST_SEEDS:
        rng = np.random.default_rng(5000 + s); order = list(rng.permutation(te_i)); phases = [order[0:10], order[10:20], order[20:30]]
        n_oos = int(round(0.125 * sum(len(R.TE[i]) for i in te_i) / 0.875)); oos = list(rng.choice(R.OOS_TEST, n_oos, replace=False))
        oos_parts = np.array_split(np.array(oos, dtype=object), 3)
        phased_t, phased_y = [], []
        for ph, op in zip(phases, oos_parts):
            items = [(t, i) for i in ph for t in R.TE[i]] + [(t, 'oos') for t in op]
            items = [items[k] for k in rng.permutation(len(items))]
            phased_t += [t for t, _ in items]; phased_y += [y for _, y in items]
        perm = rng.permutation(len(phased_t)); shuf_t = [phased_t[k] for k in perm]; shuf_y = [phased_y[k] for k in perm]
        for regime, (t, y) in {'phased': (phased_t, phased_y), 'shuffled': (shuf_t, shuf_y)}.items():
            X = enc(t)
            for m, f in methods.items():
                if regime == 'phased' and m.startswith('Revised') is None: pass
                if m.startswith('BIRCH') or m.startswith('Streaming'):
                    a, rdy, T = f(X, s)
                else:
                    out = f(X, s)
                    if m in ('Final configuration (F)', 'Revised mechanism (C)'):
                        a, rdy, mm = out
                        T = mm.mu[rdy] / (np.linalg.norm(mm.mu[rdy], axis=1, keepdims=True) + 1e-12) if len(rdy) else np.zeros((0, X.shape[1]))
                    else:
                        a, rdy, T = out
                corr, cov = correct_facts(y, a, rdy, T, cat, te_i)
                rows.append(dict(partition=part, seed=s, regime=regime, method=m,
                                 phase1=len(corr & set(phases[0])) / 10, phase2=len(corr & set(phases[1])) / 10, phase3=len(corr & set(phases[2])) / 10,
                                 overall=len(corr) / 30, neg=float(np.mean(np.isin(np.asarray(a)[np.asarray(y) == 'oos'], rdy)))))
    print('partition', part, 'done', flush=True)
    return rows


if __name__ == '__main__':
    with Pool(2) as p:
        out = p.map(job, range(5))
    json.dump([r for o in out for r in o], open('./forgetting_results.json', 'w'), default=float)
    print('done')
