"""Stage 1 with offline consolidation (replay), column splitting and support-scaled readiness, built on htm_v2.Stage1v2.
replay_every > 0 : every replay_every episodes (and at the end), the stored member episodes of all columns are
                   re-assigned to the most similar column template (if it passes the vigilance threshold tau), and
                   templates are recomputed from their stored members (a replay-based consolidation step).
split_tau        : during replay, a column whose stored members form two groups with template cosine < split_tau
                   (2-means) is split into two columns.
ready_frac > 0   : a column is READY only if its support is at least ready_frac x the mean support of the columns
                   that reach n_ready (support-scaled readiness).
mem_size         : number of member episodes stored per column (reservoir sampling)."""
import numpy as np
import htm_v2
from htm_v2 import Stage1v2

EXTRA = dict(replay_every=0, split_tau=None, ready_frac=0.0, mem_size=20)


def _n(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Stage1v3(Stage1v2):
    def __init__(self, D, var0, seed=0, **kw):
        ex = {k: kw.pop(k, v) for k, v in EXTRA.items()}
        super().__init__(D, var0, seed=seed, **kw)
        self.ex = ex; self.mem = [[] for _ in range(self.p['n_columns'])]
        self.relabel = {}; self.rng = np.random.default_rng(seed + 12345); self.ep = -1

    def _store(self, j, x):
        m = self.mem[j]
        if len(m) < self.ex['mem_size']:
            m.append((self.ep, x))
        else:
            k = self.rng.integers(0, max(self.n[j], 1))
            if k < self.ex['mem_size']:
                m[k] = (self.ep, x)

    def _merge(self):
        super()._merge()
        for j in range(len(self.mem)):
            if self.mem[j] and not self.alive[j] and self.parent[j] != j:
                r = self.find(j); self.mem[r] += self.mem[j]; self.mem[j] = []
                if len(self.mem[r]) > self.ex['mem_size']:
                    keep = self.rng.choice(len(self.mem[r]), self.ex['mem_size'], replace=False)
                    self.mem[r] = [self.mem[r][k] for k in keep]

    def _replay(self):
        p = self.p; idx = np.flatnonzero(self.alive)
        if idx.size < 2:
            return
        items = [(e, x, j) for j in idx for e, x in self.mem[j]]
        if not items:
            return
        T = _n(self.mu[idx]); X = np.array([x for _, x, _ in items]); S = X @ T.T
        best = idx[S.argmax(1)]; bs = S.max(1)
        newmem = {j: [] for j in idx}
        for (e, x, j), b, s in zip(items, best, bs):
            tgt = b if (b != j and s >= p['tau']) else j
            if tgt != j:
                self.n[j] -= 1; self.n[tgt] += 1; self.relabel[e] = tgt
            newmem[tgt].append((e, x))
        for j in idx:
            self.mem[j] = newmem[j]
            if self.mem[j]:
                self.mu[j] = np.mean([x for _, x in self.mem[j]], axis=0)
            if self.n[j] <= 0:
                self.alive[j] = False
        if self.ex['split_tau'] is not None:
            self._split()

    def _split(self):
        p = self.p
        for j in np.flatnonzero(self.alive):
            m = self.mem[j]
            if len(m) < 2 * p['n_ready']:
                continue
            X = np.array([x for _, x in m]); Sm = X @ X.T
            a, b = np.unravel_index(np.argmin(Sm), Sm.shape); c = np.stack([X[a], X[b]])
            for _ in range(5):
                lab = (X @ _n(c).T).argmax(1)
                if lab.min() == lab.max():
                    break
                c = np.stack([X[lab == 0].mean(0), X[lab == 1].mean(0)])
            if lab.min() == lab.max() or (lab == 0).sum() < p['n_ready'] or (lab == 1).sum() < p['n_ready']:
                continue
            if float(_n(c[0]) @ _n(c[1])) >= self.ex['split_tau']:
                continue
            free = np.flatnonzero(~self.used)
            if free.size == 0:
                return
            k = free[0]; self.used[k] = self.alive[k] = True
            g1 = [m[i] for i in range(len(m)) if lab[i] == 1]; g0 = [m[i] for i in range(len(m)) if lab[i] == 0]
            frac = len(g1) / len(m); nk = max(int(round(self.n[j] * frac)), len(g1))
            self.n[k] = nk; self.n[j] = max(self.n[j] - nk, len(g0))
            self.mu[k] = c[1]; self.mu[j] = c[0]; self.anchor[k] = c[1]; self.var[k] = self.var[j].copy()
            self.conf[k] = self.conf[j]; self.res[k] = [x for _, x in g1][:p['reservoir']]; self.last[k] = self.t
            self.mem[k] = g1; self.mem[j] = g0
            for e, _ in g1:
                self.relabel[e] = k

    def step(self, x):
        self.ep += 1
        j = super().step(x)
        if j >= 0:
            self._store(j, x)
        if self.ex['replay_every'] and (self.ep + 1) % self.ex['replay_every'] == 0:
            self._replay()
        return j

    def finish(self, raw_assign):
        if self.p['merge']:
            self._merge()
        if self.ex['replay_every']:
            self._replay()
        raw = [self.relabel.get(i, a) for i, a in enumerate(raw_assign)]
        out = np.array([-1 if a < 0 else self.find(a) for a in raw])
        out[np.isin(out, np.flatnonzero(self.retired))] = -1
        ok = self.alive & (self.n >= self.p['n_ready'])
        if self.ex['ready_frac'] > 0 and ok.any():
            ok &= self.n >= self.ex['ready_frac'] * self.n[ok].mean()
        ready = np.flatnonzero(ok)
        T = _n(self.mu[ready]) if ready.size else np.zeros((0, self.mu.shape[1]))
        return out, ready, T


def run_v3(X, var0, seed, **kw):
    m = Stage1v3(X.shape[1], var0, seed=seed, **kw)
    raw = [m.step(x) for x in X]
    a, r, T = m.finish(raw)
    return a, r, T, m
