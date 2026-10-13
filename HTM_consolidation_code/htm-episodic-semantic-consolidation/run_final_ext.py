"""Final configuration (F): 20 seeds per fold/partition with extended measures (as Tables A10/A11) and order vs
initialization sensitivity, using the configuration selected on validation data in run_improved.py."""
import sys, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.'); sys.path.insert(0, './synthetic')
import run_improved as RI
from run_extended import ext
from htm_v2 import run_v2
import run_clinc as R
SEL = {}
for r in json.load(open('./improved_results.json')):
    SEL[(r['dataset'], r['unit'])] = r['selected']

def job(spec):
    ds, unit = spec; D = RI.setup(ds, unit); enc = D['enc']
    Xtr = enc(D['tr_txt']); var0 = np.mean([Xtr[D['tr_lab'] == l].var(0) for l in set(D['tr_lab'])], axis=0)
    sel = dict(SEL[(ds, unit)]); rs = sel.pop('ready_sim'); kw = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in sel.items()}
    cte = enc(D['te_f']); rows = []
    if ds == 'synthetic':
        import run_synthetic as RS
        S = RS.S; te_s = [s for s in S if s['fold'] == unit]; mk = lambda s: RS.stream(te_s, 'test', s)
    else:
        rng = np.random.default_rng(1000 + unit); perm = rng.permutation(R.INTENTS); te_i = perm[120:]
        mk = lambda s: R.make_stream(te_i, R.TE, R.OOS_TEST, s)
    def one(t, y, seed):
        a, rdy, T, m = run_v2(enc(t), var0, seed, **kw)
        rdy = RI.compact_ready(m, rdy, rs)
        T = m.mu[rdy] / (np.linalg.norm(m.mu[rdy], axis=1, keepdims=True) + 1e-12) if len(rdy) else np.zeros((0, m.mu.shape[1]))
        return ext(y, a, rdy, T, cte, D['te_n'])
    t1, y1 = mk(1)
    for s in range(1, 21):
        t, y = mk(s)
        rows.append(dict(dataset=ds, unit=unit, seed=s, kind='main', **one(t, y, s)))
        rows.append(dict(dataset=ds, unit=unit, seed=s, kind='order', **one(t, y, 0)))
        rows.append(dict(dataset=ds, unit=unit, seed=s, kind='init', **one(t1, y1, s)))
    return rows

if __name__ == '__main__':
    specs = [('synthetic', f) for f in range(5)] + [('CLINC150', p) for p in range(5)]
    with Pool(2) as p:
        out = p.map(job, specs)
    json.dump([r for o in out for r in o], open('./final_ext_results.json', 'w'), default=float)
    print('done')
