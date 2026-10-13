"""Reviewer 1, Comment 1.8 on CLINC150 with the revised mechanism (configuration C).
 (1) fitting:   encoder statistics fitted on training-intent text only  vs  on all text incl. test utterances
                and test fact names (transductive, the procedure used for the synthetic experiments)
 (2) decoding:  closed-set catalogue of the 30 test facts  vs  an enlarged catalogue of all 150 facts
 (3) supervision: weakly supervised (contrastive projection) vs fully unsupervised (no projection)
Thresholds are selected on validation intents for each encoder setting; results on test intents, 5 partitions x 5 seeds."""
import json, itertools, time
import numpy as np
from multiprocessing import Pool
import run_clinc as R
from htm_consolidation import Encoder, train_contrastive
from htm_v2 import GloveEncoder, run_v2

GRID = [dict(routing='similarity', merge=True, tau=t, tau_merge=m, z_thresh=np.inf)
        for t, m in itertools.product([0.25, 0.35, 0.45, 0.55, 0.65], [0.75, 0.85, 0.92])]


def build(kind, fit_texts, tr, lab, part, contrastive):
    enc = (Encoder(seed=part) if kind == 'TF-IDF+SVD' else GloveEncoder()).fit(fit_texts)
    if contrastive:
        enc.proj = train_contrastive(enc.base(tr), list(lab), seed=part)
    return enc


def one(part):
    t0 = time.time(); rows = []
    rng = np.random.default_rng(1000 + part); perm = rng.permutation(R.INTENTS)
    tr_i, va_i, te_i = perm[:90], perm[90:120], perm[120:]
    tr = [t for i in tr_i for t in R.TR[i]]; lab = np.array([i for i in tr_i for _ in R.TR[i]])
    all_text = ([t for i in R.INTENTS for s in (R.TR, R.VA, R.TE) for t in s[i]] + R.OOS_VAL + R.OOS_TEST
                + [R.fact_text(i) for i in R.INTENTS])
    settings = [  # (encoder, fit scope, contrastive)
        ('GloVe', 'train only', True), ('GloVe', 'all text (transductive)', True),
        ('TF-IDF+SVD', 'train only', True), ('TF-IDF+SVD', 'all text (transductive)', True),
        ('GloVe', 'train only', False)]
    val_streams = [R.make_stream(va_i, R.VA, R.OOS_VAL, s) for s in R.VAL_SEEDS]
    for kind, scope, con in settings:
        enc = build(kind, tr if scope == 'train only' else all_text, tr, lab, part, con)
        Xtr = enc(tr); var0 = np.mean([Xtr[lab == i].var(0) for i in tr_i], axis=0)
        Xv = [(enc(t), y) for t, y in val_streams]; cv = enc([R.fact_text(i) for i in va_i])
        best, bs = None, -1
        for kw in GRID:
            sc = np.mean([R.evaluate(y, *run_v2(X, var0, s, **kw)[:3], cv, list(va_i))['BcF1'] for (X, y), s in zip(Xv, R.VAL_SEEDS)])
            if sc > bs:
                bs, best = sc, kw
        cat_test = enc([R.fact_text(i) for i in te_i]); cat_all = enc([R.fact_text(i) for i in R.INTENTS])
        for s in R.TEST_SEEDS:
            txt, y = R.make_stream(te_i, R.TE, R.OOS_TEST, s); X = enc(txt)
            a, rdy, T, m = run_v2(X, var0, s, **best)
            for catname, cat, names in [('closed set: 30 test facts', cat_test, list(te_i)), ('enlarged: all 150 facts', cat_all, list(R.INTENTS))]:
                if catname.startswith('enlarged') and not (kind == 'GloVe' and scope == 'train only' and con):
                    continue
                rows.append(dict(partition=part, seed=s, encoder=kind, fit=scope, contrastive=con, catalogue=catname,
                                 tau=best['tau'], tau_merge=best['tau_merge'], **R.evaluate(y, a, rdy, T, cat, names)))
        print(part, kind, scope, con, best['tau'], best['tau_merge'], f'{time.time()-t0:.0f}s', flush=True)
    return rows


if __name__ == '__main__':
    with Pool(2) as p:
        out = p.map(one, range(R.N_PARTITIONS))
    json.dump([r for o in out for r in o], open('./leakage_results.json', 'w'), indent=1, default=float)
    print('done')
