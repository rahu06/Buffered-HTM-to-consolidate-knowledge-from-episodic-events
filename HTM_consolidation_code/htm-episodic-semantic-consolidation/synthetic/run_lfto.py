"""
Leave-fact-type-out (LFTO) validation of the contrastive projection (Reviewer 1, Comment 1.9), revised mechanism (C).
For test fold f (validation fold f+1, three training folds) and each positive fact type t:
  - LFTO projection: encoder statistics and contrastive projection fitted on training-fold samples EXCLUDING type t;
    thresholds selected on the validation fold EXCLUDING type t (no supervision or selection signal from type t);
  - compared with the projection trained on all types (standard protocol) and with no projection (unsupervised).
Reported: fact accuracy on the held-out type's test facts (the whole test stream is processed, all types + negatives).
"""
import sys, json, itertools, time
import numpy as np
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, '.'); sys.path.insert(0, './synthetic')
import run_synthetic as RS
import run_clinc as R
from htm_consolidation import train_contrastive
from htm_v2 import GloveEncoder, run_v2

S = RS.S
TYPES = sorted({s['fact_type'] for s in S if s['target_fact']})
GRID = [RS.base_kw(routing='similarity', merge=True, tau=t, tau_merge=m, z_thresh=np.inf)
        for t, m in itertools.product([0.55, 0.65, 0.75, 0.85, 0.9], [0.85, 0.92, 0.96])]


def correct_facts(y, assign, ready, T, cat, names):
    y = np.asarray(y); assign = np.asarray(assign)
    if not len(ready):
        return set()
    dec = [names[i] for i in np.argmax(T @ cat.T, axis=1)]
    ok = set()
    for c, dd in zip(ready, dec):
        lab = Counter(y[assign == c]).most_common(1)[0][0]
        if lab == dd:
            ok.add(lab)
    return ok


def make_enc(tr_samples, contrastive, seed):
    txt, lab, _ = RS.episodes(tr_samples, 'train')
    enc = GloveEncoder().fit(txt)
    if contrastive:
        enc.proj = train_contrastive(enc.base(txt), list(lab), steps=600, batch=min(128, len(set(lab))), seed=seed)
    X = enc(txt)
    var0 = np.mean([X[lab == l].var(0) for l in set(lab)], axis=0)
    return enc, var0


def select(enc, var0, va_samples):
    facts = [s['target_fact'] for s in va_samples if s['target_fact']]
    cat = enc(facts)
    streams = [RS.stream(va_samples, 'val', s) for s in RS.SEEDS[:3]]
    return max(GRID, key=lambda kw: np.mean([R.evaluate(y, *run_v2(enc(t), var0, s, **kw)[:3], cat, facts)['BcF1'] for (t, y), s in zip(streams, RS.SEEDS[:3])]))


def test_per_type(enc, var0, kw, te_samples, f, tag, only_type=None):
    rows = []
    facts = [s['target_fact'] for s in te_samples if s['target_fact']]
    cat = enc(facts)
    by_type = {t: {s['target_fact'] for s in te_samples if s['fact_type'] == t} for t in TYPES}
    for s in RS.SEEDS:
        t_, y = RS.stream(te_samples, 'test', s)
        a, rdy, T, m = run_v2(enc(t_), var0, s, **kw)
        ok = correct_facts(y, a, rdy, T, cat, facts)
        for ty in ([only_type] if only_type else TYPES):
            rows.append(dict(fold=f, seed=s, setting=tag, held_out_type=ty, fact_accuracy=len(ok & by_type[ty]) / len(by_type[ty])))
    return rows


def one(f):
    t0 = time.time(); rows = []
    v = (f + 1) % 5
    tr = [s for s in S if s['fold'] not in (f, v) and s['target_fact']]
    va = [s for s in S if s['fold'] == v]; te = [s for s in S if s['fold'] == f]
    # standard protocol: projection on all types; and no projection
    for tag, con in [('Projection trained on all fact types', True), ('No projection (unsupervised)', False)]:
        enc, var0 = make_enc(tr, con, f)
        rows += test_per_type(enc, var0, select(enc, var0, va), te, f, tag)
    # leave-fact-type-out
    for ty in TYPES:
        tr_t = [s for s in tr if s['fact_type'] != ty]
        va_t = [s for s in va if s['fact_type'] != ty]
        enc, var0 = make_enc(tr_t, True, f)
        rows += test_per_type(enc, var0, select(enc, var0, va_t), te, f, 'Projection trained without the held-out type (LFTO)', only_type=ty)
    print(f'fold {f} done {time.time()-t0:.0f}s', flush=True)
    return rows


if __name__ == '__main__':
    with Pool(2) as p:
        out = p.map(one, range(5))
    json.dump([r for o in out for r in o], open('./synthetic/lfto_results.json', 'w'), indent=1)
    print('done')
