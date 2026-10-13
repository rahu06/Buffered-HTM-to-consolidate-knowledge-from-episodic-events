import json, numpy as np
import run_clinc as R
from htm_consolidation import Encoder, train_contrastive
res = json.load(open('clinc_results.json'))
sel = {(r['partition'], r['encoder']): (r['tau'], r['z']) for r in res if r['method'].startswith('HTM columns (full')}
out = []
for part in range(R.N_PARTITIONS):
    rng = np.random.default_rng(1000 + part); perm = rng.permutation(R.INTENTS)
    tr_i, te_i = perm[:90], perm[120:]
    tr_text = [t for i in tr_i for t in R.TR[i]]; tr_lab = np.array([i for i in tr_i for _ in R.TR[i]])
    enc = Encoder(seed=part).fit(tr_text); Xtr = enc.base(tr_text)
    enc.proj = train_contrastive(Xtr, list(tr_lab), seed=part); Xtr = enc.proj(Xtr)
    var0 = np.mean([Xtr[tr_lab == i].var(0) for i in tr_i], axis=0)
    tau, z = sel[(part, 'TF-IDF+SVD + contrastive')]
    for s in R.TEST_SEEDS:
        txt, y = R.make_stream(te_i, R.TE, R.OOS_TEST, s); X = enc(txt); cat = enc([R.fact_text(i) for i in te_i])
        for name, kw in [('shortlist = all columns', dict(candidate_k=1024)),
                         ('shortlist = all, tau=0.35 (manuscript value)', dict(candidate_k=1024, tau=0.35, anchor_tau=0.35, z_thresh=1.0))]:
            kk = dict(tau=tau, anchor_tau=tau, z_thresh=z); kk.update(kw)
            a, rdy, T, m = R.run_stage1(X, var0, s, **kk)
            out.append(dict(method=name, **R.evaluate(y, a, rdy, T, cat, list(te_i))))
for n in sorted({o['method'] for o in out}):
    rs = [o for o in out if o['method'] == n]
    print(n, {k: f"{np.mean([r[k] for r in rs]):.3f}±{np.std([r[k] for r in rs]):.2f}" for k in ['FactAccuracy','ColumnExactMatch','Purity','BcF1','ARI','NMI','Coverage','NegAbsorbed']})
json.dump(out, open('sens_results.json','w'), default=float)
