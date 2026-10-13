"""Further continual-learning tests on CLINC150 (Reviewer 1 Comment 1.23; Reviewer 2 Comments 2.1, 2.3, 2.4).
(a) skewed streams: the number of episodes per test fact follows a Zipf law (n_r = max(3, round(30 / r)), random ranks);
    fact accuracy overall, for the 10 most frequent (head) and the 10 least frequent (tail) facts.
(b) long streams: all 150 utterances of each test intent (train, val and test splits of CLINC150; about 5,100 episodes
    with negatives), presented in three phases of ten intents; old-fact retention as in the forgetting test.
(c) abrupt drift: at the middle of the stream the embedding space is rotated at once to cosine c in {0.9, 0.8, 0.7};
    the catalogue is embedded with the end-of-stream encoder.
(d) transfer and plasticity on the standard phased stream: accuracy on phase-k facts right after phase k (learning
    accuracy), at the end (retention), backward transfer BWT = mean_k<3 [acc_end(k) - acc_after_k(k)], and forward
    transfer FWT = mean_k>1 [acc_after_k(k) in the sequence - acc on phase k presented alone].
Settings: those selected on validation data in the main experiments. 5 partitions x 5 streams."""
import sys, json
import numpy as np
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, '.')
import run_clinc as R, run_improved as RI, new_baselines as NB
from htm_v2 import run_v2
from run_drift import rotate

ONLY_SKEW = len(sys.argv) > 1 and sys.argv[1] == 'skew'
IMP = {r['unit']: r['selected'] for r in json.load(open('./improved_results.json')) if r['dataset'] == 'CLINC150'}
C120 = [r for r in json.load(open('./c120_results_final.json')) if r['part'] == 'baselines' and r['dataset'] == 'CLINC150']


def correct_facts(y, assign, ready, T, cat, names):
    y = np.asarray(y); a = np.asarray(assign).copy(); nxt = a.max() + 1 if len(a) else 0
    for k in np.flatnonzero(a < 0): a[k] = nxt; nxt += 1
    if not len(ready): return set()
    dec = [names[i] for i in np.argmax(T @ cat.T, axis=1)]
    corr = set()
    for c, d in zip(ready, dec):
        lab = Counter(y[a == c]).most_common(1)[0][0]
        if lab != 'oos' and d == lab: corr.add(lab)
    return corr


def job(part):
    D = RI.setup('CLINC150', part); enc = D['enc']
    Xtr = enc(D['tr_txt']); var0 = np.mean([Xtr[D['tr_lab'] == l].var(0) for l in set(D['tr_lab'])], axis=0)
    lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
    te_i = list(D['te_n']); cat0 = enc(D['te_f'])
    selF = dict(IMP[part]); rsF = selF.pop('ready_sim'); kwF = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in selF.items()}
    kwF['n_columns'] = 2048
    par = {r['method']: r.get('param') for r in C120 if r['unit'] == part}

    def F(X, s):
        a, rdy, T, m = run_v2(X, var0, s, **kwF); rdy = RI.compact_ready(m, rdy, rsF)
        return a, rdy, (m.mu[rdy] / (np.linalg.norm(m.mu[rdy], axis=1, keepdims=True) + 1e-12) if len(rdy) else np.zeros((0, X.shape[1])))
    methods = {
        'Final configuration (F)': F,
        'BIRCH (online)': lambda X, s: R.run_baseline('BIRCH (online)', X, 30, s, par['BIRCH (online)']),
        'Fuzzy ART': lambda X, s: R.clusters_to_ready(X, NB.fuzzy_art(X, par['Fuzzy ART'], lo, hi)),
        'DenStream': lambda X, s: R.clusters_to_ready(X, NB.denstream(X, par['DenStream'])),
        'Online nearest-prototype memory': lambda X, s: R.clusters_to_ready(X, NB.leader_memory(X, par['Online nearest-prototype memory'])),
    }
    rows = []
    for s in R.TEST_SEEDS:
        rng = np.random.default_rng(8000 + 10 * part + s)
        # (a) skewed stream
        ranks = rng.permutation(len(te_i)); n_r = {te_i[k]: max(3, int(round(30 / (ranks[k] + 1)))) for k in range(len(te_i))}
        items = [(t, i) for i in te_i for t in rng.choice(R.TE[i], n_r[i], replace=False)]
        n_oos = int(round(0.125 * len(items) / 0.875)); items += [(t, 'oos') for t in rng.choice(R.OOS_TEST, n_oos, replace=False)]
        order = rng.permutation(len(items)); t = [items[k][0] for k in order]; y = [items[k][1] for k in order]; X = enc(t)
        head = {te_i[k] for k in range(len(te_i)) if ranks[k] < 10}; tail = {te_i[k] for k in range(len(te_i)) if ranks[k] >= 20}
        for m, f in methods.items():
            a, rdy, T = f(X, s); corr = correct_facts(y, a, rdy, T, cat0, te_i)
            ev = R.evaluate(y, a, rdy, T, cat0, te_i)
            rows.append(dict(test='skewed', partition=part, seed=s, method=m, acc=len(corr) / 30, head=len(corr & head) / len(head),
                             tail=len(corr & tail) / len(tail), neg=ev['NegAbsorbed'], n_eps=len(X)))
        if ONLY_SKEW: continue
        # phases (shared by b and d)
        phases = [list(x) for x in np.array_split(rng.permutation(te_i), 3)]
        def phased(pool_fn, oos_n_total):
            oos = list(rng.choice(R.OOS_TEST, oos_n_total, replace=False)); oparts = np.array_split(np.array(oos, dtype=object), 3)
            tt, yy, bounds = [], [], []
            for ph, op in zip(phases, oparts):
                it = [(u, i) for i in ph for u in pool_fn(i)] + [(u, 'oos') for u in op]
                it = [it[k] for k in rng.permutation(len(it))]
                tt += [u for u, _ in it]; yy += [l for _, l in it]; bounds.append(len(tt))
            return tt, yy, bounds
        # (b) long stream
        long_pool = lambda i: list(R.TR[i]) + list(R.VA[i]) + list(R.TE[i])
        n_pos = sum(len(long_pool(i)) for i in te_i); tt, yy, bounds = phased(long_pool, min(len(R.OOS_TEST), int(round(0.125 * n_pos / 0.875))))
        X = enc(tt); perm = rng.permutation(len(tt)); Xs = X[perm]; ys = [yy[k] for k in perm]
        for m, f in methods.items():
            a, rdy, T = f(X, s); c_ph = correct_facts(yy, a, rdy, T, cat0, te_i)
            a2, r2, T2 = f(Xs, s); c_sh = correct_facts(ys, a2, r2, T2, cat0, te_i)
            ev = R.evaluate(yy, a, rdy, T, cat0, te_i)
            rows.append(dict(test='long', partition=part, seed=s, method=m, n_eps=len(tt), old=len(c_ph & set(phases[0])) / len(phases[0]),
                             recent=len(c_ph & set(phases[2])) / len(phases[2]), all=len(c_ph) / 30, old_control=len(c_sh & set(phases[0])) / len(phases[0]),
                             neg=ev['NegAbsorbed']))
        # (c) abrupt drift on the standard test stream
        t0_, y0 = D['test'][R.TEST_SEEDS.index(s)]; X0 = enc(t0_); n = len(X0)
        U, _ = np.linalg.qr(np.random.default_rng(7700 + 10 * part + s).standard_normal((X0.shape[1], X0.shape[1])))
        for c_end in (1.0, 0.9, 0.8, 0.7):
            th = np.where(np.arange(n) >= n // 2, np.arccos(c_end), 0.0)
            Xd = rotate(X0, th, U); Xd /= np.linalg.norm(Xd, axis=1, keepdims=True) + 1e-12
            cat = rotate(cat0, np.full(len(cat0), np.arccos(c_end)), U)
            for m, f in methods.items():
                a, rdy, T = f(Xd, s); corr = correct_facts(y0, a, rdy, T, cat, te_i)
                half2 = set(np.asarray(y0)[n // 2:]) - {'oos'}
                rows.append(dict(test='abrupt', partition=part, seed=s, method=m, c_end=c_end, acc=len(corr) / 30))
        # (d) transfer and plasticity on the standard phased stream
        tt, yy, bounds = phased(lambda i: list(R.TE[i]), int(round(0.125 * 900 / 0.875)))
        X = enc(tt)
        for m, f in methods.items():
            after = []
            for k, b in enumerate(bounds):
                a, rdy, T = f(X[:b], s); corr = correct_facts(yy[:b], a, rdy, T, cat0, te_i)
                after.append({j: len(corr & set(phases[j])) / len(phases[j]) for j in range(k + 1)})
            alone = []
            for k in (1, 2):
                st = bounds[k - 1]; Xk = X[st:bounds[k]]; yk = yy[st:bounds[k]]
                a, rdy, T = f(Xk, s); corr = correct_facts(yk, a, rdy, T, cat0, te_i); alone.append(len(corr & set(phases[k])) / len(phases[k]))
            learn = [after[k][k] for k in range(3)]
            bwt = np.mean([after[2][k] - after[k][k] for k in range(2)])
            fwt = np.mean([after[k][k] - alone[k - 1] for k in (1, 2)])
            rows.append(dict(test='transfer', partition=part, seed=s, method=m, learning_acc=float(np.mean(learn)), final_acc=float(np.mean([after[2][k] for k in range(3)])),
                             bwt=float(bwt), fwt=float(fwt)))
    print('partition', part, 'done', flush=True)
    return rows


if __name__ == '__main__':
    with Pool(1 if ONLY_SKEW else 2) as p:
        out = p.map(job, range(5))
    json.dump([r for o in out for r in o], open('./cl2_skew.json' if ONLY_SKEW else './cl2_results.json', 'w'), default=float)
    print('done')
