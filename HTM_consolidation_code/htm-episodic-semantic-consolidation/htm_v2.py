"""
Improved Stage 1 (v2) with switchable changes, plus a pretrained-embedding encoder.
  routing='similarity'  : shortlist by cosine to column templates instead of random projection
  merge=True            : periodic merge of near-duplicate mature columns (offline consolidation)
  adaptive=True         : per-column vigilance from the spread of its own members' similarities
  shrink_n0>0           : precision gate with variance shrinkage toward a global variance
  retire_after>0        : release columns that stay below readiness for too long
All switches off reproduces the v1 reimplementation (htm_consolidation.Stage1).
"""
import re
import numpy as np

# ---------------------------------------------------------------- GloVe encoder
_G = None
def _glove():
    global _G
    if _G is None:
        d = np.load('./glove_clinc.npy', allow_pickle=True).item()
        _G = dict(zip(d['words'], d['vecs']))
    return _G

tok = lambda t: re.findall(r"[a-z0-9']+", t.lower())

class GloveEncoder:
    """IDF-weighted mean of pretrained GloVe vectors (Wikipedia), common component removed (SIF),
    L2-normalized. IDF and the common component are fitted on training-intent text only."""
    def __init__(self):
        self.proj = None

    def fit(self, texts):
        G = _glove(); n = len(texts); df = {}
        for t in texts:
            for w in set(tok(t)):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        self.idf_unk = np.log(1 + n) + 1
        E = self._raw(texts)
        u, s, vt = np.linalg.svd(E - 0, full_matrices=False)
        self.pc = vt[0]
        return self

    def _raw(self, texts):
        G = _glove(); out = np.zeros((len(texts), 100))
        for k, t in enumerate(texts):
            ws = [w for w in tok(t) if w in G]
            if ws:
                wt = np.array([self.idf.get(w, self.idf_unk) for w in ws])
                out[k] = (wt[:, None] * np.stack([G[w] for w in ws])).sum(0) / wt.sum()
        return out

    def base(self, texts):
        E = self._raw(texts)
        E = E - np.outer(E @ self.pc, self.pc)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    def __call__(self, texts):
        z = self.base(texts)
        return self.proj(z) if self.proj is not None else z


# ---------------------------------------------------------------- Stage 1 v2
V2 = dict(n_columns=1024, candidate_k=20, tau=0.35, anchor_tau=None, z_thresh=1.0, eps=1e-6,
          beta=0.30, alpha_min=0.05, lam=0.8, reservoir=5, precision_min_support=2, n_ready=3,
          weighting='confidence', routing='random',
          merge=False, tau_merge=0.85, merge_every=50,
          adaptive=False, k_sd=2.0, tau_floor_gap=0.2,
          shrink_n0=0, retire_after=0, use_anchor=True, use_reservoir=True, ready_sim=None)


def _n(v):
    return v / (np.linalg.norm(v) + 1e-12)


class Stage1v2:
    def __init__(self, D, var0, seed=0, **kw):
        p = dict(V2, **kw)
        if p['anchor_tau'] is None:
            p['anchor_tau'] = p['tau']
        self.p = p; N = p['n_columns']
        self.M = np.random.default_rng(seed).normal(size=(N, D))
        self.mu = np.zeros((N, D)); self.var = np.zeros((N, D)); self.anchor = np.zeros((N, D))
        self.conf = np.zeros(N); self.n = np.zeros(N, int)
        self.alive = np.zeros(N, bool); self.used = np.zeros(N, bool)
        self.parent = np.arange(N); self.retired = np.zeros(N, bool)
        self.res = [[] for _ in range(N)]
        self.sm = np.zeros(N); self.s2 = np.zeros(N); self.sn = np.zeros(N)   # member-similarity stats
        self.last = np.zeros(N, int); self.t = 0
        self.var0 = var0; self.vglob = var0.copy(); self.dropped = 0

    def find(self, j):
        while self.parent[j] != j:
            self.parent[j] = self.parent[self.parent[j]]; j = self.parent[j]
        return j

    def _thr(self, j):
        p = self.p
        if p['adaptive'] and self.sn[j] >= 3:
            sd = np.sqrt(max(self.s2[j] / self.sn[j] - (self.sm[j] / self.sn[j]) ** 2, 0))
            t = self.sm[j] / self.sn[j] - p['k_sd'] * sd
            return float(np.clip(t, p['tau'] - p['tau_floor_gap'], 0.95))
        return p['tau']

    def _gates(self, x, j, sim):
        p = self.p; thr = self._thr(j)
        if sim <= thr:
            return False
        a_thr = thr if p['adaptive'] else p['anchor_tau']
        if p['use_anchor'] and x @ self.anchor[j] <= a_thr - (0.1 if p['adaptive'] else 0):
            return False
        if p['use_reservoir'] and self.res[j] and np.mean([x @ r for r in self.res[j]]) <= a_thr - (0.1 if p['adaptive'] else 0):
            return False
        if self.n[j] >= p['precision_min_support'] and np.isfinite(p['z_thresh']):
            v = self.var[j]
            if p['shrink_n0'] > 0:
                v = (self.n[j] * v + p['shrink_n0'] * self.vglob) / (self.n[j] + p['shrink_n0'])
            if np.sqrt(np.mean((x - self.mu[j]) ** 2 / (v + p['eps']))) >= p['z_thresh']:
                return False
        return True

    def _merge(self):
        p = self.p
        idx = np.flatnonzero(self.alive & (self.n >= 2))
        if idx.size < 2:
            return
        T = self.mu[idx] / (np.linalg.norm(self.mu[idx], axis=1, keepdims=True) + 1e-12)
        S = T @ T.T; np.fill_diagonal(S, -1)
        pairs = np.argwhere(np.triu(S, 1) > p['tau_merge'])
        for a, b in sorted(pairs.tolist(), key=lambda ab: -S[ab[0], ab[1]]):
            i, j = idx[a], idx[b]
            if not (self.alive[i] and self.alive[j]):
                continue
            if self.n[j] > self.n[i]:
                i, j = j, i
            ni, nj = self.n[i], self.n[j]; m = (ni * self.mu[i] + nj * self.mu[j]) / (ni + nj)
            self.var[i] = (ni * (self.var[i] + (self.mu[i] - m) ** 2) + nj * (self.var[j] + (self.mu[j] - m) ** 2)) / (ni + nj)
            self.mu[i] = m; self.n[i] = ni + nj; self.conf[i] = max(self.conf[i], self.conf[j])
            self.res[i] = (self.res[i] + self.res[j])[:p['reservoir']]
            self.sm[i] += self.sm[j]; self.s2[i] += self.s2[j]; self.sn[i] += self.sn[j]
            self.last[i] = max(self.last[i], self.last[j])
            self.alive[j] = False; self.parent[j] = i

    def _retire(self):
        p = self.p
        old = self.alive & (self.n < p['n_ready']) & (self.t - self.last > p['retire_after'])
        self.alive[old] = False; self.retired[old] = True

    def step(self, x):
        p = self.p; self.t += 1
        if p['merge'] and self.t % p['merge_every'] == 0:
            self._merge()
        if p['retire_after'] and self.t % 50 == 0:
            self._retire()
        idx = np.flatnonzero(self.alive); chosen = -1
        if idx.size:
            T = self.mu[idx] / (np.linalg.norm(self.mu[idx], axis=1, keepdims=True) + 1e-12)
            sims_all = T @ x
            if p['routing'] == 'similarity':
                order = np.argsort(-sims_all)[:p['candidate_k']]
            else:
                cand = np.argsort(-(self.M[idx] @ x))[:p['candidate_k']]
                order = cand[np.argsort(-sims_all[cand])]
            for o in order:
                if self._gates(x, idx[o], sims_all[o]):
                    chosen = idx[o]; s_acc = sims_all[o]; break
        if chosen < 0:
            free = np.flatnonzero(~self.used)
            if free.size == 0:
                self.dropped += 1; return -1
            j = free[0]; self.used[j] = self.alive[j] = True
            self.anchor[j] = x; self.mu[j] = x.copy(); self.var[j] = self.var0.copy()
            self.conf[j] = 0; self.n[j] = 1; self.res[j] = [x]; self.last[j] = self.t
            return j
        j = chosen; d = x - self.mu[j]
        self.sm[j] += s_acc; self.s2[j] += s_acc ** 2; self.sn[j] += 1
        self.var[j] = (1 - p['beta']) * self.var[j] + p['beta'] * d * d
        eta = 1.0 / (self.n[j] + 1) if p['weighting'] == 'mean' else (1 - self.conf[j]) * (1 - p['alpha_min']) + p['alpha_min']
        self.mu[j] = self.mu[j] + eta * d
        self.conf[j] = p['lam'] * self.conf[j] + (1 - p['lam']) * np.exp(-self.var[j].mean())
        self.n[j] += 1; self.last[j] = self.t
        self.res[j].append(x)
        if len(self.res[j]) > p['reservoir']:
            self.res[j].pop(1)
        return j

    def finish(self, raw_assign):
        if self.p['merge']:
            self._merge()
        out = np.array([-1 if a < 0 else self.find(a) for a in raw_assign])
        out[np.isin(out, np.flatnonzero(self.retired))] = -1
        ready = np.flatnonzero(self.alive & (self.n >= self.p['n_ready']))
        if self.p['ready_sim'] is not None and ready.size:   # compactness: members must agree closely with the template
            keep = []
            for j in ready:
                Tj = self.mu[j] / (np.linalg.norm(self.mu[j]) + 1e-12)
                if np.mean([r @ Tj for r in self.res[j]]) >= self.p['ready_sim']:
                    keep.append(j)
            ready = np.array(keep, dtype=int)
        T = self.mu[ready] / (np.linalg.norm(self.mu[ready], axis=1, keepdims=True) + 1e-12) if ready.size else np.zeros((0, self.mu.shape[1]))
        return out, ready, T


def run_v2(X, var0, seed, **kw):
    m = Stage1v2(X.shape[1], var0, seed=seed, **{k: v for k, v in kw.items() if k != 'kappa'})
    raw = [m.step(x) for x in X]
    a, r, T = m.finish(raw)
    if kw.get('kappa') is not None:
        a, r, T = verify(X, a, r, T, kw['kappa'], m.p['n_ready'])
    return a, r, T, m


def verify(X, assign, ready, T, kappa, n_ready):
    """Consolidation-time verification: an episode stays in a READY column only if its cosine similarity to the
    column's final template is at least kappa; released episodes become unconsolidated, and a column remains READY
    only if at least n_ready verified episodes support it (its template is recomputed from them)."""
    a = np.array(assign).copy()
    if len(ready) == 0:
        return a, ready, T
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    keep_r, keep_T = [], []
    for c, t in zip(ready, T):
        idx = np.flatnonzero(a == c)
        bad = idx[Xn[idx] @ t < kappa]
        a[bad] = -1
        good = np.setdiff1d(idx, bad)
        if good.size >= n_ready:
            mu = X[good].mean(0); keep_r.append(c); keep_T.append(mu / (np.linalg.norm(mu) + 1e-12))
    return a, np.array(keep_r, int), (np.array(keep_T) if keep_T else np.zeros((0, X.shape[1])))
