"""
External validation of the Stage 1 reimplementation on CLINC150 (Larson et al., 2019).
Protocol (Reviewer 1, Comment 1.7): intent-disjoint train / validation / test splits.
  * train intents: fit TF-IDF+SVD and the contrastive projection (train utterances only)
  * validation intents: select thresholds (no test data used)
  * test intents: report results; facts (intents) never seen by the encoder or projection
Out-of-scope (oos) utterances act as negative controls, as the no_generalization class does.
"""
import json, itertools, sys, time
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch, AgglomerativeClustering, DBSCAN
from htm_consolidation import Encoder, train_contrastive, run_stage1 as _rs
N_COLUMNS = 1024   # pool scaled to the ~1,000-episode stream (manuscript: 256 for 254 episodes)
def run_stage1(X, var0, seed, **kw):
    kw.setdefault('n_columns', N_COLUMNS)
    return _rs(X, var0, seed, **kw)

D = json.load(open('./data_full.json'))
INTENTS = sorted({l for _, l in D['train']})
N_PARTITIONS, TEST_SEEDS, VAL_SEEDS = 5, [1, 7, 42, 123, 999], [1, 7, 42]
N_TRAIN_I, N_VAL_I, N_TEST_I = 90, 30, 30
OOS_FRAC = 0.125                        # ~10/80 negatives, as in the original dataset


def by_intent(split):
    d = defaultdict(list)
    for t, l in D[split]:
        d[l].append(t)
    return d

TR, VA, TE = by_intent('train'), by_intent('val'), by_intent('test')
OOS_VAL = [t for t, _ in D['oos_val']]
OOS_TEST = [t for t, _ in D['oos_test']]
fact_text = lambda i: i.replace('_', ' ')


def make_stream(intents, pool, oos, seed):
    rng = np.random.default_rng(seed)
    texts = [(t, i) for i in intents for t in pool[i]]
    n_oos = int(round(OOS_FRAC * len(texts) / (1 - OOS_FRAC)))
    texts += [(t, 'oos') for t in rng.choice(oos, n_oos, replace=False)]
    order = rng.permutation(len(texts))
    return [texts[k][0] for k in order], np.array([texts[k][1] for k in order])


# ---------------------------------------------------------------- metrics
def bcubed(y, a):
    pos = {}
    for k, (yy, aa) in enumerate(zip(y, a)):
        pos.setdefault(aa, []).append(k)
    P = R = 0.0
    yc = Counter(y)
    for k, (yy, aa) in enumerate(zip(y, a)):
        members = pos[aa]
        same = sum(1 for m in members if y[m] == yy)
        P += same / len(members); R += same / yc[yy]
    P /= len(y); R /= len(y)
    return P, R, 2 * P * R / (P + R)


def evaluate(y, assign, ready, templates, cat_emb, cat_names):
    y = np.asarray(y); assign = np.asarray(assign).copy()
    # unassigned episodes become singletons
    nxt = assign.max() + 1 if assign.size else 0
    for k in np.flatnonzero(assign < 0):
        assign[k] = nxt; nxt += 1
    pos = y != 'oos'
    intents = sorted(set(y[pos]))
    out = {}
    out['ARI'] = adjusted_rand_score(y[pos], assign[pos])
    out['NMI'] = normalized_mutual_info_score(y[pos], assign[pos])
    out['BcP'], out['BcR'], out['BcF1'] = bcubed(list(y[pos]), list(assign[pos]))
    decoded = [cat_names[i] for i in np.argmax(templates @ cat_emb.T, axis=1)] if len(ready) else []
    maj, pur, ok = [], [], []
    for c, dec in zip(ready, decoded):
        mem = y[assign == c]
        lab, cnt = Counter(mem).most_common(1)[0]
        maj.append(lab); pur.append(cnt / len(mem)); ok.append(dec == lab)
    nr = max(len(ready), 1)
    out['READY'] = len(ready)
    out['ColumnExactMatch'] = sum(ok) / nr                      # manuscript definition
    out['Purity'] = float(np.mean(pur)) if pur else 0.0
    out['Contamination'] = sum(m == 'oos' for m in maj) / nr    # manuscript "false-positive rate"
    in_ready = np.isin(assign, ready)
    out['NegAbsorbed'] = float(in_ready[~pos].mean()) if (~pos).any() else 0.0  # FP/(FP+TN) over negatives
    covered = {m for m in maj if m != 'oos'}
    correct = {m for m, k in zip(maj, ok) if k}
    out['Coverage'] = len(covered) / len(intents)
    out['FactAccuracy'] = len(correct) / len(intents)          # fact-level: penalizes missed facts
    out['Fragmentation'] = sum(m != 'oos' for m in maj) / max(len(covered), 1)
    return out


def clusters_to_ready(X, labels, min_size=3):
    labels = np.asarray(labels).copy()
    ready, T = [], []
    for c in np.unique(labels):
        if c < 0:
            continue
        idx = labels == c
        if idx.sum() >= min_size:
            ready.append(c); m = X[idx].mean(0); T.append(m / (np.linalg.norm(m) + 1e-12))
    nxt = labels.max() + 1
    for k in np.flatnonzero(labels < 0):
        labels[k] = nxt; nxt += 1
    return labels, np.array(ready), (np.array(T) if T else np.zeros((0, X.shape[1])))


# ---------------------------------------------------------------- baselines
def run_baseline(name, X, k_true, seed, param=None):
    X = X.copy(); z = np.linalg.norm(X, axis=1) < 1e-9
    X[z] = 1e-6          # all-OOV utterances (zero TF-IDF vector); cosine undefined otherwise
    if name == 'K-Means (oracle k)':
        lab = KMeans(k_true, n_init=10, random_state=seed).fit_predict(X)
    elif name == 'Streaming mini-batch K-Means (oracle k)':
        mb = MiniBatchKMeans(k_true, random_state=seed, batch_size=32, n_init=3)
        for s in range(0, len(X), 32):
            chunk = X[s:s + 32]
            if s == 0 and len(chunk) < k_true:
                continue
            mb.partial_fit(chunk)
        lab = mb.predict(X)
    elif name == 'BIRCH (online)':
        b = Birch(threshold=param, n_clusters=None)
        for s in range(0, len(X), 32):
            b.partial_fit(X[s:s + 32])
        lab = b.predict(X)
    elif name == 'Agglomerative (cosine)':
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=param, metric='cosine', linkage='average').fit_predict(X)
    elif name == 'DBSCAN (cosine)':
        lab = DBSCAN(eps=param, min_samples=3, metric='cosine').fit_predict(X)
    return clusters_to_ready(X, lab)

BASE_GRID = {'BIRCH (online)': [0.3, 0.4, 0.5, 0.6, 0.7],
             'Agglomerative (cosine)': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
             'DBSCAN (cosine)': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]}
GRID = list(itertools.product([0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.45, 0.55, 0.65], [1.0, 1.5, 2.0, np.inf]))


def main():
    rows = []
    t0 = time.time()
    for part in range(N_PARTITIONS):
        rng = np.random.default_rng(1000 + part)
        perm = rng.permutation(INTENTS)
        tr_i, va_i, te_i = perm[:N_TRAIN_I], perm[N_TRAIN_I:N_TRAIN_I + N_VAL_I], perm[N_TRAIN_I + N_VAL_I:]
        assert not (set(tr_i) & set(va_i) or set(tr_i) & set(te_i) or set(va_i) & set(te_i))
        tr_text = [t for i in tr_i for t in TR[i]]
        tr_lab = [i for i in tr_i for _ in TR[i]]
        for enc_name in ['TF-IDF+SVD', 'TF-IDF+SVD + contrastive']:
            enc = Encoder(seed=part).fit(tr_text)               # train intents only
            Xtr = enc.base(tr_text)
            if enc_name.endswith('contrastive'):
                enc.proj = train_contrastive(Xtr, tr_lab, seed=part)
                Xtr = enc.proj(Xtr)
            lab_arr = np.array(tr_lab)
            var0 = np.mean([Xtr[lab_arr == i].var(0) for i in tr_i], axis=0)  # within-fact variance, train intents
            # ---- threshold selection on validation intents
            best, best_score = None, -1
            for tau, z in GRID:
                sc = []
                for s in VAL_SEEDS:
                    txt, y = make_stream(va_i, VA, OOS_VAL, s)
                    X = enc(txt)
                    a, rdy, T, _ = run_stage1(X, var0, s, tau=tau, anchor_tau=tau, z_thresh=z)
                    cat = enc([fact_text(i) for i in va_i])
                    sc.append(evaluate(y, a, rdy, T, cat, list(va_i))['BcF1'])
                if np.mean(sc) > best_score:
                    best_score, best = np.mean(sc), (tau, z)
            bparam = {}
            for bn, grid in (BASE_GRID.items() if enc_name.endswith('contrastive') else []):
                bs, bv = -1, None
                for g in grid:
                    sc = []
                    for s in VAL_SEEDS:
                        txt, y = make_stream(va_i, VA, OOS_VAL, s)
                        X = enc(txt)
                        lab, rdy, T = run_baseline(bn, X, len(va_i), s, g)
                        sc.append(evaluate(y, lab, rdy, T, enc([fact_text(i) for i in va_i]), list(va_i))['BcF1'])
                    if np.mean(sc) > bs:
                        bs, bv = np.mean(sc), g
                bparam[bn] = bv
            tau, z = best
            print(f'partition {part} {enc_name}: selected tau={tau} z={z}  ({time.time()-t0:.0f}s)', flush=True)
            # ---- test intents (never used for fitting or selection)
            cat_names = list(te_i)
            for s in TEST_SEEDS:
                txt, y = make_stream(te_i, TE, OOS_TEST, s)
                X = enc(txt)
                cat = enc([fact_text(i) for i in te_i])
                configs = {
                    'HTM columns (full gating, confidence-weighted)': dict(tau=tau, anchor_tau=tau, z_thresh=z),
                    'Same pipeline, unweighted mean pooling': dict(tau=tau, anchor_tau=tau, z_thresh=z, weighting='mean'),
                    'Same pipeline, no precision gate': dict(tau=tau, anchor_tau=tau, z_thresh=np.inf),
                    'HTM columns, 256-column pool (sensitivity)': dict(tau=tau, anchor_tau=tau, z_thresh=z, n_columns=256),
                }
                for cn, kw in configs.items():
                    a, rdy, T, m = run_stage1(X, var0, s, **kw)
                    r = evaluate(y, a, rdy, T, cat, cat_names)
                    rows.append(dict(partition=part, seed=s, encoder=enc_name, method=cn, tau=tau, z=z, dropped=m.dropped, **r))
                if enc_name.endswith('contrastive'):
                    for bn in ['K-Means (oracle k)', 'Streaming mini-batch K-Means (oracle k)', 'BIRCH (online)', 'Agglomerative (cosine)', 'DBSCAN (cosine)']:
                        lab, rdy, T = run_baseline(bn, X, len(te_i), s, bparam.get(bn))
                        r = evaluate(y, lab, rdy, T, cat, cat_names)
                        rows.append(dict(partition=part, seed=s, encoder=enc_name, method=bn, param=bparam.get(bn), **r))
    json.dump(rows, open('./clinc_results.json', 'w'), indent=1, default=float)
    print('done', len(rows), f'{time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
