"""
Old vs revised consolidation mechanism on the synthetic benchmark v2 (episodic_semantic_dataset_v2.json).
Protocol: 5-fold cross-validation with entity-disjoint folds; for test fold f, fold f+1 is the validation fold and the
other three folds are training folds. Training text uses the 'train' template set, validation the 'val' set and test
the 'test' set, so sentence templates never overlap. Encoders and the contrastive projection are fitted on training
folds only (except the explicitly transductive rows), thresholds are selected on the validation fold, and results are
reported on the test fold; 5 seeds per fold (stream order and random projection) -> 25 runs per setting.
"""
import sys, json, itertools, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
import run_clinc as R
import htm_v2
from htm_consolidation import Encoder, train_contrastive
from htm_v2 import GloveEncoder, run_v2

g = np.load('./synthetic/glove_synthetic.npy', allow_pickle=True).item()
htm_v2._G = dict(zip(g['words'], g['vecs']))

DATA = json.load(open('./synthetic/episodic_semantic_dataset_v2.json'))
S = DATA['samples']
SEEDS = [1, 7, 42, 123, 999]
ALL_FACTS = [s['target_fact'] for s in S if s['target_fact']]
N_READY = 2   # samples have 3-4 episodes; a fact needs at least two consistent episodes


def episodes(samples, split):
    txt, lab, sid = [], [], []
    for s in samples:
        for e in s['episodes']:
            txt.append(e['text'][split]); lab.append(s['target_fact'] or 'oos'); sid.append(s['sample_id'])
    return txt, np.array(lab), sid


def stream(samples, split, seed):
    txt, lab, _ = episodes(samples, split)
    o = np.random.default_rng(seed).permutation(len(txt))
    return [txt[k] for k in o], lab[o]


def base_kw(**kw):
    return dict(n_columns=256, n_ready=N_READY, **kw)


A_GRID = [dict(tau=t, z_thresh=z) for t, z in itertools.product([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95], [1.0, 2.0, np.inf])]
def grid(extra):
    keys = list(extra)
    for t, z, *v in itertools.product([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95], [1.0, 2.0, np.inf], *[extra[k] for k in keys]):
        yield dict(tau=t, z_thresh=z, **dict(zip(keys, v)))

ABL = {'A. Base configuration (random shortlist)': (dict(), {}),
       'B. + similarity routing': (dict(routing='similarity'), {}),
       'C. + column merging (revised mechanism)': (dict(routing='similarity', merge=True), {'tau_merge': [0.75, 0.85, 0.92, 0.96]}),
       'D. + adaptive vigilance': (dict(routing='similarity', merge=True, adaptive=True), {'tau_merge': [0.75, 0.85, 0.92, 0.96], 'k_sd': [1.5, 2.5]}),
       'E. + shrinkage gate and retirement': (dict(routing='similarity', merge=True, adaptive=True, shrink_n0=5, retire_after=60), {'tau_merge': [0.75, 0.85, 0.92, 0.96], 'k_sd': [1.5, 2.5]})}
BGRID = {'BIRCH (online)': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'Agglomerative (cosine)': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
         'DBSCAN (cosine)': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]}
BASELINES = ['K-Means (true number of facts)', 'Streaming mini-batch K-Means (true number of facts)', 'BIRCH (online)', 'Agglomerative (cosine)', 'DBSCAN (cosine)']
BMAP = {'K-Means (true number of facts)': 'K-Means (oracle k)', 'Streaming mini-batch K-Means (true number of facts)': 'Streaming mini-batch K-Means (oracle k)'}


def ready2(X, labels):
    return R.clusters_to_ready(X, labels, min_size=N_READY)


def baseline(bn, X, k, s, p):
    name = BMAP.get(bn, bn)
    X2 = X.copy(); X2[np.linalg.norm(X2, axis=1) < 1e-9] = 1e-6
    import sklearn.cluster as C
    if name == 'K-Means (oracle k)':
        lab = C.KMeans(k, n_init=10, random_state=s).fit_predict(X2)
    elif name == 'Streaming mini-batch K-Means (oracle k)':
        mb = C.MiniBatchKMeans(k, random_state=s, batch_size=32, n_init=3)
        first = max(k, 32)
        mb.partial_fit(X2[:first])
        for st in range(first, len(X2), 32):
            mb.partial_fit(X2[st:st + 32])
        lab = mb.predict(X2)
    elif name == 'BIRCH (online)':
        b = C.Birch(threshold=p, n_clusters=None)
        for st in range(0, len(X2), 32):
            b.partial_fit(X2[st:st + 32])
        lab = b.predict(X2)
    elif name == 'Agglomerative (cosine)':
        lab = C.AgglomerativeClustering(n_clusters=None, distance_threshold=p, metric='cosine', linkage='average').fit_predict(X2)
    else:
        lab = C.DBSCAN(eps=p, min_samples=N_READY, metric='cosine').fit_predict(X2)
    return ready2(X, lab)


def select(enc, var0, val, va_facts, fixed, extra):
    cat = enc(va_facts); best, bs = None, -1
    for gkw in grid(extra):
        kw = base_kw(**fixed, **gkw)
        sc = np.mean([R.evaluate(y, *run_v2(enc(t), var0, s, **kw)[:3], cat, va_facts)['BcF1'] for (t, y), s in zip(val, SEEDS[:3])])
        if sc > bs:
            bs, best = sc, kw
    return best


def one(f):
    t0 = time.time(); rows = []
    v = (f + 1) % 5
    tr_s = [s for s in S if s['fold'] not in (f, v)]; va_s = [s for s in S if s['fold'] == v]; te_s = [s for s in S if s['fold'] == f]
    tr_pos = [s for s in tr_s if s['target_fact']]
    tr_txt, tr_lab, _ = episodes(tr_pos, 'train')
    va_facts = [s['target_fact'] for s in va_s if s['target_fact']]
    te_facts = [s['target_fact'] for s in te_s if s['target_fact']]
    val = [stream(va_s, 'val', s) for s in SEEDS[:3]]
    all_text = [e['text'][k] for s in S for e in s['episodes'] for k in ('train', 'val', 'test')] + ALL_FACTS

    def make(kind, fit_scope, contrastive):
        fit = tr_txt if fit_scope == 'train' else all_text
        enc = (Encoder(seed=f) if kind == 'TF-IDF+SVD' else GloveEncoder()).fit(fit)
        if contrastive:
            enc.proj = train_contrastive(enc.base(tr_txt), list(tr_lab), steps=600, batch=min(128, len(set(tr_lab))), seed=f)
        Xtr = enc(tr_txt)
        var0 = np.mean([Xtr[tr_lab == l].var(0) for l in set(tr_lab)], axis=0)
        return enc, var0

    def test(enc, var0, kw, tag, enlarged=False):
        cat = enc(te_facts)
        for s in SEEDS:
            t, y = stream(te_s, 'test', s); X = enc(t)
            a, rdy, T, m = run_v2(X, var0, s, **kw)
            rows.append(dict(fold=f, seed=s, setting=tag, catalogue='closed set (test-fold facts)', params={k: (None if isinstance(x, float) and not np.isfinite(x) else x) for k, x in kw.items()}, **R.evaluate(y, a, rdy, T, cat, te_facts)))
            if enlarged:
                rows.append(dict(fold=f, seed=s, setting=tag, catalogue='enlarged (all 240 facts)', **R.evaluate(y, a, rdy, T, enc(ALL_FACTS), ALL_FACTS)))

    # ---- old pipeline as in the manuscript: TF-IDF+SVD, contrastive projection, base configuration, fixed thresholds
    fixed_old = base_kw(tau=0.35, z_thresh=1.0)
    enc, var0 = make('TF-IDF+SVD', 'all', True);   test(enc, var0, fixed_old, 'Old pipeline, TF-IDF fitted on all text (as in manuscript)')
    enc, var0 = make('TF-IDF+SVD', 'train', True); test(enc, var0, fixed_old, 'Old pipeline, TF-IDF fitted on training folds only')
    test(enc, var0, select(enc, var0, val, va_facts, {}, {}), 'Old pipeline, training-fold fitting, thresholds selected on validation fold')
    test(enc, var0, select(enc, var0, val, va_facts, dict(routing='similarity', merge=True), {'tau_merge': [0.75, 0.85, 0.92, 0.96]}), 'Revised mechanism with TF-IDF+SVD encoder')
    # ---- ablation with the pretrained encoder (training-fold fitting)
    enc, var0 = make('GloVe', 'train', True)
    chosen = {}
    for name, (fixed, extra) in ABL.items():
        kw = select(enc, var0, val, va_facts, fixed, extra); chosen[name] = kw
        test(enc, var0, kw, name, enlarged=name.startswith('C.'))
    # baselines on the same encoder
    cv = enc(va_facts)
    for bn in BASELINES:
        p = None
        if bn in BGRID:
            p = max(BGRID[bn], key=lambda q: np.mean([R.evaluate(y, *baseline(bn, enc(t), len(va_facts), s, q), cv, va_facts)['BcF1'] for (t, y), s in zip(val, SEEDS[:3])]))
        cat = enc(te_facts)
        for s in SEEDS:
            t, y = stream(te_s, 'test', s); X = enc(t)
            rows.append(dict(fold=f, seed=s, setting=bn, catalogue='closed set (test-fold facts)', params={'param': p}, **R.evaluate(y, *baseline(bn, X, len(te_facts), s, p), cat, te_facts)))
    # ---- revised mechanism: transductive fitting and fully unsupervised variants
    Ckw = chosen['C. + column merging (revised mechanism)']
    enc2, var2 = make('GloVe', 'all', True); test(enc2, var2, select(enc2, var2, val, va_facts, dict(routing='similarity', merge=True), {'tau_merge': [0.75, 0.85, 0.92, 0.96]}), 'Revised mechanism, encoder fitted on all text (transductive)')
    enc3, var3 = make('GloVe', 'train', False); test(enc3, var3, select(enc3, var3, val, va_facts, dict(routing='similarity', merge=True), {'tau_merge': [0.75, 0.85, 0.92, 0.96]}), 'Revised mechanism, fully unsupervised (no contrastive projection)')
    print(f'fold {f} done {time.time()-t0:.0f}s', {k[:2]: (v_['tau'], v_.get('tau_merge'), v_['z_thresh']) for k, v_ in chosen.items()}, flush=True)
    return rows


if __name__ == '__main__':
    with Pool(2) as p:
        out = p.map(one, range(5))
    json.dump([r for o in out for r in o], open('./synthetic/synthetic_results.json', 'w'), indent=1, default=float)
    print('done')
