"""Reviewer 1 Comment 1.7 / Reviewer 3 Comment 3.3: Stage 1 as a memory module on LoCoMo (10 long conversations).
Episodes: the session observations of LoCoMo (2,541 short statements about the two speakers, each linked to the
dialogue turn that supports it). The stream of observations of one conversation, in session order, is consolidated
online by Stage 1 (final configuration: similarity routing, running mean, merging). The memory then holds the READY
columns (each pointing to its member observations) and the observations that were not consolidated.
Retrieval test: for each question of categories 1-4 (multi-hop, temporal, open-domain, single-hop; adversarial
questions excluded), memory items are ranked by cosine similarity to the question, and observations are read from the
ranked items until a budget of B observations is reached (inside a column, members are ranked by similarity to the
question). Evidence recall@B = fraction of the question's evidence turns covered. References: the raw episodic store
(every observation is an item) and the online nearest-prototype memory. The vigilance threshold is selected by
two-fold cross-validation over conversations (0-4 / 5-9) on recall@10. Encoder: SIF-weighted GloVe-100, fitted
without labels on the observations of the conversation."""
import gzip, json, re, sys
import numpy as np
sys.path.insert(0, '.')
import htm_v2
from htm_v2 import GloveEncoder, run_v2
import new_baselines as NB

D = json.load(open('./data/locomo10.json'))
tok = lambda t: re.findall(r"[a-z0-9']+", t.lower())
need = set()
for c in D:
    for s in c['observation'].values():
        for v in s.values():
            for t, _ in v: need |= set(tok(t))
    for q in c['qa']: need |= set(tok(q['question']))
G = {}
with gzip.open('./glove100.gz', 'rt', encoding='utf8') as f:
    for line in f:
        w, rest = line.split(' ', 1)
        if w in need: G[w] = np.array(rest.split(), dtype=np.float32)
htm_v2._G = G
BUDGETS = (5, 10, 20)


def conv_data(c):
    obs = []
    keys = sorted(c['observation'], key=lambda k: int(re.findall(r'\d+', k)[0]))
    for k in keys:
        for spk, v in c['observation'][k].items():
            for t, ev in v:
                ids = ev if isinstance(ev, list) else [ev]
                obs.append((t, set(i.strip() for i in ids for i in str(i).split(';'))))
    qs = [q for q in c['qa'] if q['category'] in (1, 2, 3, 4) and q.get('evidence')]
    return obs, qs


def items_from_groups(X, groups):
    """groups: list of index lists -> item vectors (normalized means) and members."""
    V = np.array([X[g].mean(0) / (np.linalg.norm(X[g].mean(0)) + 1e-12) for g in groups])
    return V, groups


def retrieve(qv, V, groups, X, B):
    order = np.argsort(-(V @ qv)); out = []
    for k in order:
        mem = sorted(groups[k], key=lambda i: -(X[i] @ qv))
        out += mem
        if len(out) >= B: break
    return out[:B]


def memory(method, X, tau):
    n = len(X)
    if method == 'Raw episodic store':
        return [[i] for i in range(n)]
    if method == 'Stage 1 (final configuration)':
        var0 = np.full(X.shape[1], X.var(0).mean())
        a, rdy, T, m = run_v2(X, var0, 0, n_columns=1024, n_ready=2, routing='similarity', tau=tau, weighting='mean',
                              use_anchor=False, use_reservoir=False, z_thresh=np.inf, merge=True, tau_merge=0.85)
        a = np.asarray(a); groups = [list(np.flatnonzero(a == c)) for c in rdy]
        inr = set(i for g in groups for i in g)
        return groups + [[i] for i in range(n) if i not in inr]
    if method == 'Online nearest-prototype memory':
        lab = np.asarray(NB.leader_memory(X, tau))
        return [list(np.flatnonzero(lab == l)) for l in sorted(set(lab))]


def run_conv(ci, method, tau):
    obs, qs = conv_data(D[ci])
    texts = [t for t, _ in obs]
    enc = GloveEncoder().fit(texts); X = enc(texts); Q = enc([q['question'] for q in qs])
    groups = memory(method, X, tau); V, groups = items_from_groups(X, groups)
    res = []
    for qv, q in zip(Q, qs):
        ev = set(q['evidence']); r = {}
        for B in BUDGETS:
            got = set().union(*[obs[i][1] for i in retrieve(qv, V, groups, X, B)])
            r[B] = len(ev & got) / len(ev)
        res.append(dict(conv=ci, cat=q['category'], **{f'recall@{B}': r[B] for B in BUDGETS}))
    return res, len(groups), len(obs)


if __name__ == '__main__':
    out = []
    TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]
    for method in ('Raw episodic store', 'Stage 1 (final configuration)', 'Online nearest-prototype memory'):
        for fold in ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)):
            other = [i for i in range(10) if i not in fold]
            tau = 0.0
            if method != 'Raw episodic store':
                tau = max(TAUS, key=lambda t: np.mean([r['recall@10'] for i in other for r in run_conv(i, method, t)[0]]))
            for ci in fold:
                res, n_items, n_obs = run_conv(ci, method, tau)
                for r in res: r.update(method=method, tau=tau, n_items=n_items, n_obs=n_obs)
                out += res
        print(method, 'done', flush=True)
    json.dump(out, open('./locomo_results.json', 'w'), default=float)
    import collections
    for m in ('Raw episodic store', 'Stage 1 (final configuration)', 'Online nearest-prototype memory'):
        rs = [r for r in out if r['method'] == m]
        items = np.mean(list({(r['conv'], r['n_items']) for r in rs}), axis=0)[1]
        print(m, 'items/conv %.0f' % items, ' '.join('R@%d %.3f' % (B, np.mean([r[f'recall@{B}'] for r in rs])) for B in BUDGETS),
              ' | by cat R@10:', {c: round(np.mean([r['recall@10'] for r in rs if r['cat'] == c]), 3) for c in (1, 2, 3, 4)})
