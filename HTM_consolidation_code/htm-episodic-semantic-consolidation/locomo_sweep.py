import sys, json, numpy as np
sys.argv = ['x']
sys.path.insert(0, '.')
import run_locomo as L
out = []
for method in ('Stage 1 (final configuration)', 'Online nearest-prototype memory'):
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        rs, items, obs = [], [], []
        for ci in range(10):
            r, n_items, n_obs = L.run_conv(ci, method, tau); rs += r; items.append(n_items); obs.append(n_obs)
        row = dict(method=method, tau=tau, memory_fraction=float(np.sum(items) / np.sum(obs)),
                   **{f'R@{B}': float(np.mean([x[f'recall@{B}'] for x in rs])) for B in L.BUDGETS},
                   **{f'R@10_cat{c}': float(np.mean([x['recall@10'] for x in rs if x['cat'] == c])) for c in (1, 2, 3, 4)})
        print(row, flush=True); out.append(row)
json.dump(out, open('./locomo_sweep.json', 'w'))
