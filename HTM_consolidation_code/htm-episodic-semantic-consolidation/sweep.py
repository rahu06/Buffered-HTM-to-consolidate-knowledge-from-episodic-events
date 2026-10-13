import json, numpy as np
import run_clinc as R
from htm_consolidation import Encoder, train_contrastive
out = []
for part in range(R.N_PARTITIONS):
    rng = np.random.default_rng(1000 + part); perm = rng.permutation(R.INTENTS)
    tr_i, te_i = perm[:90], perm[120:]
    tr_text = [t for i in tr_i for t in R.TR[i]]; tr_lab = np.array([i for i in tr_i for _ in R.TR[i]])
    enc = Encoder(seed=part).fit(tr_text); Xtr = enc.base(tr_text)
    enc.proj = train_contrastive(Xtr, list(tr_lab), seed=part); Xtr = enc.proj(Xtr)
    var0 = np.mean([Xtr[tr_lab == i].var(0) for i in tr_i], axis=0)
    for s in R.TEST_SEEDS:
        txt, y = R.make_stream(te_i, R.TE, R.OOS_TEST, s); X = enc(txt); cat = enc([R.fact_text(i) for i in te_i])
        for tau in [0.25, 0.35, 0.45, 0.55]:
            for wname, w in [('HTM columns (confidence-weighted)', 'confidence'), ('Same pipeline, mean pooling', 'mean')]:
                a, rdy, T, m = R.run_stage1(X, var0, s, candidate_k=1024, tau=tau, anchor_tau=tau, z_thresh=1.0, weighting=w)
                out.append(dict(part=part, seed=s, method=wname, param=tau, **R.evaluate(y, a, rdy, T, cat, list(te_i))))
        for bn, grid in [('Agglomerative (cosine)', [0.3, 0.4, 0.5, 0.6, 0.7]), ('BIRCH (online)', [0.5, 0.6, 0.7, 0.8, 0.9]), ('DBSCAN (cosine)', [0.1, 0.15, 0.2, 0.3])]:
            for g in grid:
                lab, rdy, T = R.run_baseline(bn, X, len(te_i), s, g)
                out.append(dict(part=part, seed=s, method=bn, param=g, **R.evaluate(y, lab, rdy, T, cat, list(te_i))))
json.dump(out, open('sweep_results.json', 'w'), default=float)
from collections import defaultdict
g = defaultdict(list)
for o in out: g[(o['method'], o['param'])].append(o)
ks = ['Purity', 'NegAbsorbed', 'Fragmentation', 'FactAccuracy', 'ColumnExactMatch', 'Coverage', 'BcF1', 'READY']
print(f"{'method':36s}{'param':>6s}" + ''.join(f'{k[:10]:>11s}' for k in ks))
for (m, p), rs in g.items():
    print(f"{m:36s}{p:6.2f}" + ''.join(f"{np.mean([r[k] for r in rs]):11.3f}" for k in ks))
