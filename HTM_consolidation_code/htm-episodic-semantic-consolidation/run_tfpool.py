"""Reviewer 2, Comment 2.2: buffer-enhanced/column consolidation vs transformer-based pooling (accuracy and cost), CLINC150.
Transformer pooling (TP): a 2-layer transformer encoder (4 heads, width 128) over the set of episode embeddings of one
group, followed by attention pooling, trained on the training intents (90 intents, never the test intents) to map a
group of 3-30 episodes, optionally contaminated with out-of-scope episodes, to the embedding of its fact text
(InfoNCE over the facts in the batch). Untrained self-attention pooling and mean pooling (the running-mean template of
the mechanism) are the references.
(1) Oracle grouping: each test fact's episodes (group sizes 3, 5, 10, 30; clean or with 20% negative episodes) are
    pooled and decoded against the 30 test facts.
(2) Online consolidation: the columns formed online by the final configuration are decoded either from their
    running-mean templates (as in the paper) or from transformer pooling of their member episodes.
Cost: trainable parameters, training time, and pooling time per episode (CPU, one thread)."""
import sys, json, time
import numpy as np, torch, torch.nn as nn
from multiprocessing import Pool
sys.path.insert(0, '.')
import run_clinc as R, run_improved as RI
from htm_v2 import run_v2
torch.set_num_threads(1)
IMP = {r['unit']: r['selected'] for r in json.load(open('./improved_results.json')) if r['dataset'] == 'CLINC150'}


class TP(nn.Module):
    def __init__(self, D, d=128):
        super().__init__()
        self.inp = nn.Linear(D, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 256, 0.1, batch_first=True), 2)
        self.q = nn.Parameter(torch.randn(d) * 0.02); self.out = nn.Linear(d, D)

    def forward(self, X, m):                       # X: B x N x D, m: B x N (True = present)
        h = self.enc(self.inp(X), src_key_padding_mask=~m)
        s = (h @ self.q).masked_fill(~m, -1e9); w = torch.softmax(s, 1)
        z = self.out((w.unsqueeze(-1) * h).sum(1))
        return nn.functional.normalize(z, dim=-1)


def self_attn_pool(G):                             # untrained: softmax(G G^T) G, then mean
    A = np.exp(G @ G.T / 0.1); A /= A.sum(1, keepdims=True); z = (A @ G).mean(0); return z / np.linalg.norm(z)


def sample_group(Xi, Xoos, size, noise, rng):
    k = min(size, len(Xi)); G = Xi[rng.choice(len(Xi), k, replace=False)]
    if noise > 0:
        n = max(1, int(round(noise * k / (1 - noise)))); G = np.vstack([G, Xoos[rng.choice(len(Xoos), n, replace=False)]])
    return G[rng.permutation(len(G))]


def pad(groups, D, N=40):
    X = np.zeros((len(groups), N, D), np.float32); m = np.zeros((len(groups), N), bool)
    for i, g in enumerate(groups):
        g = g[:N]; X[i, :len(g)] = g; m[i, :len(g)] = True
    return torch.tensor(X), torch.tensor(m)


def job(part):
    torch.manual_seed(part); rng = np.random.default_rng(part)
    D = RI.setup('CLINC150', part); enc = D['enc']
    perm = np.random.default_rng(1000 + part).permutation(R.INTENTS); tr_i = list(perm[:90])
    Xtr = {i: enc(R.TR[i]) for i in tr_i}; Ftr = enc([R.fact_text(i) for i in tr_i]); Xoos_tr = enc(R.OOS_VAL)
    te_i = list(D['te_n']); Xte = {i: enc(R.TE[i]) for i in te_i}; Fte = enc(D['te_f']); Xoos = enc(list(R.OOS_TEST[:300]))
    Dm = Ftr.shape[1]
    model = TP(Dm); opt = torch.optim.Adam(model.parameters(), 1e-3)
    n_par = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    for step in range(1500):
        idx = rng.choice(len(tr_i), 32, replace=False)
        groups = [sample_group(Xtr[tr_i[k]], Xoos_tr, int(rng.choice([3, 5, 10, 30])), float(rng.choice([0.0, 0.2])), rng) for k in idx]
        X, m = pad(groups, Dm); z = model(X, m); F = torch.tensor(Ftr[idx], dtype=torch.float32)
        loss = nn.functional.cross_entropy(z @ F.T / 0.05, torch.arange(len(idx)))
        opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0
    model.eval(); rows = []
    for size in (3, 5, 10, 30):
        for noise in (0.0, 0.2):
            for s in range(5):
                r2 = np.random.default_rng(100 * part + 10 * size + s + int(noise * 1000))
                groups = [sample_group(Xte[i], Xoos, size, noise, r2) for i in te_i]
                truth = np.arange(len(te_i))
                mean_z = np.array([g.mean(0) / np.linalg.norm(g.mean(0)) for g in groups])
                sa_z = np.array([self_attn_pool(g) for g in groups])
                t1 = time.perf_counter()
                with torch.no_grad():
                    X, m = pad(groups, Dm); tp_z = model(X, m).numpy()
                tp_time = (time.perf_counter() - t1) / sum(len(g) for g in groups) * 1000
                t1 = time.perf_counter(); _ = [g.mean(0) for g in groups]; mean_time = (time.perf_counter() - t1) / sum(len(g) for g in groups) * 1000
                for name, Z in (('Mean pooling (running-mean template)', mean_z), ('Self-attention pooling (untrained)', sa_z), ('Transformer pooling (trained)', tp_z)):
                    acc = float(np.mean(np.argmax(Z @ Fte.T, 1) == truth))
                    rows.append(dict(part='oracle', partition=part, size=size, noise=noise, rep=s, method=name, acc=acc,
                                     ms_per_episode=tp_time if 'Transformer' in name else mean_time))
    # (2) online consolidation: mechanism columns decoded from running mean vs transformer pooling of members
    sel = dict(IMP[part]); rs = sel.pop('ready_sim'); kw = {k: (np.inf if (k == 'z_thresh' and v is None) else v) for k, v in sel.items()}
    var0 = np.mean([Xtr[i].var(0) for i in tr_i], 0)
    for s, (t, y) in zip(R.TEST_SEEDS, D['test']):
        X = enc(t); t1 = time.perf_counter(); a, rdy, T, m = run_v2(X, var0, s, **kw)
        rdy = RI.compact_ready(m, rdy, rs); mech_time = (time.perf_counter() - t1) / len(X) * 1000
        Tm = m.mu[rdy] / (np.linalg.norm(m.mu[rdy], axis=1, keepdims=True) + 1e-12)
        ev = R.evaluate(y, a, rdy, Tm, Fte, te_i)
        groups = [X[np.asarray(a) == c] for c in rdy]
        t1 = time.perf_counter()
        with torch.no_grad():
            Zt = np.vstack([model(*pad([g], Dm)).numpy() for g in groups]) if len(groups) else np.zeros((0, Dm))
        tp_time = (time.perf_counter() - t1) / len(X) * 1000
        ev2 = R.evaluate(y, a, rdy, Zt, Fte, te_i)
        rows.append(dict(part='online', partition=part, seed=s, method='Mechanism, running-mean templates', acc=ev['FactAccuracy'], ms_per_episode=mech_time))
        rows.append(dict(part='online', partition=part, seed=s, method='Mechanism columns, transformer-pooled templates', acc=ev2['FactAccuracy'], ms_per_episode=mech_time + tp_time))
    rows.append(dict(part='cost', partition=part, params=n_par, train_s=train_time))
    print('partition', part, 'done', round(train_time), flush=True)
    return rows


if __name__ == '__main__':
    with Pool(2) as p:
        out = p.map(job, range(5))
    json.dump([r for o in out for r in o], open('./tfpool_results.json', 'w'), default=float)
    print('done')
