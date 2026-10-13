"""
Reimplementation of Stage 1 (online episodic-to-semantic consolidation) of
"Buffer-Enhanced HTM Columns for Episodic-to-Semantic Knowledge Consolidation",
written from the revised manuscript (Sections 3.2-3.5, Eqs. 1-6, Algorithm 1).
It is NOT the authors' original code; parameters the manuscript leaves open are
set as documented in DEFAULTS and tuned on validation intents only.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------------- encoder
class Encoder:
    """TF-IDF -> truncated SVD (D=64) -> L2, optional frozen residual projection.
    Fitted on training-intent text only (no test text, no labels)."""
    def __init__(self, D=64, seed=0):
        self.tfidf = TfidfVectorizer()
        self.svd = TruncatedSVD(n_components=D, random_state=seed)
        self.proj = None

    def fit(self, texts):
        self.svd.fit(self.tfidf.fit_transform(texts))
        return self

    def base(self, texts):
        z = self.svd.transform(self.tfidf.transform(texts))
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    def __call__(self, texts):
        z = self.base(texts)
        if self.proj is not None:
            z = self.proj(z)
        return z


def train_contrastive(X, y, hidden=128, steps=600, batch=128, lr=1e-3, temp=0.1, seed=0):
    """Residual MLP z = normalize(x + W2 relu(W1 x + b1)), InfoNCE with in-batch
    negatives; positive = another episode with the same label (Section 4.5)."""
    import autograd.numpy as anp
    from autograd import grad
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    params = [rng.normal(0, 0.05, (D, hidden)), np.zeros(hidden), rng.normal(0, 0.05, (hidden, D))]

    def f(p, x):
        h = anp.maximum(0, anp.dot(x, p[0]) + p[1])
        o = x + anp.dot(h, p[2])
        return o / (anp.sqrt(anp.sum(o * o, axis=1, keepdims=True)) + 1e-12)

    def loss(p, a, b):
        za, zb = f(p, a), f(p, b)
        s = anp.dot(za, zb.T) / temp
        s = s - anp.max(s, axis=1, keepdims=True)
        return -anp.mean(anp.diag(s) - anp.log(anp.sum(anp.exp(s), axis=1)))

    g = grad(loss)
    by = {}
    for i, c in enumerate(y):
        by.setdefault(c, []).append(i)
    labels = list(by)
    m = [np.zeros_like(q) for q in params]; v = [np.zeros_like(q) for q in params]
    for t in range(1, steps + 1):
        cls = rng.choice(labels, size=min(batch, len(labels)), replace=False)  # one pair per class -> clean negatives
        ia, ib = [], []
        for c in cls:
            a, b = rng.choice(by[c], 2, replace=False)
            ia.append(a); ib.append(b)
        gr = g(params, X[ia], X[ib])
        for k in range(3):  # Adam
            m[k] = 0.9 * m[k] + 0.1 * gr[k]; v[k] = 0.999 * v[k] + 0.001 * gr[k] ** 2
            params[k] = params[k] - lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    P = [np.array(q) for q in params]
    return lambda x: np.asarray(f(P, x))


# ---------------------------------------------------------------- Stage 1
DEFAULTS = dict(
    n_columns=256, candidate_k=20,          # Section 3.3 / Table A8
    tau=0.35, anchor_tau=0.35,              # Section 5.9 (final)
    z_thresh=1.0, eps=1e-6,                 # Eq. 6, Section 5.2
    beta=0.30,                              # Eq. 3
    alpha_min=0.05, lam=0.8,                # Eqs. 2, 4 (not given in manuscript)
    reservoir=5,                            # reservoir size R (not given)
    precision_min_support=2,                # precision gate used once variance is estimated
    n_ready=3,                              # readiness support (not given)
    weighting='confidence',                 # 'confidence' (Eqs. 1-2) or 'mean' (unweighted pooling)
)


class Stage1:
    def __init__(self, D, var0, seed=0, **kw):
        self.p = dict(DEFAULTS, **kw)
        N = self.p['n_columns']
        rng = np.random.default_rng(seed)
        self.M = rng.normal(size=(N, D))      # fixed random projection (shortlist)
        self.mu = np.zeros((N, D)); self.var = np.zeros((N, D)); self.anchor = np.zeros((N, D))
        self.conf = np.zeros(N); self.n = np.zeros(N, int)
        self.committed = np.zeros(N, bool)
        self.res = [[] for _ in range(N)]
        self.var0 = var0
        self.dropped = 0

    def _gates(self, x, j):
        p = self.p
        mu = self.mu[j] / (np.linalg.norm(self.mu[j]) + 1e-12)
        if x @ mu <= p['tau']:
            return False
        if x @ self.anchor[j] <= p['anchor_tau']:
            return False
        if self.res[j] and np.mean([x @ r for r in self.res[j]]) <= p['anchor_tau']:
            return False
        if self.n[j] >= p['precision_min_support'] and np.isfinite(p['z_thresh']):
            z = np.sqrt(np.mean((x - self.mu[j]) ** 2 / (self.var[j] + p['eps'])))
            if z >= p['z_thresh']:
                return False
        return True

    def step(self, x):
        """Process one episode embedding; return the column index it joined (or -1)."""
        p = self.p
        idx = np.flatnonzero(self.committed)
        chosen = -1
        if idx.size:
            act = self.M[idx] @ x
            cand = idx[np.argsort(-act)[:p['candidate_k']]]
            # ART-style search: candidates tried in order of similarity to their mean
            sims = np.array([x @ (self.mu[j] / (np.linalg.norm(self.mu[j]) + 1e-12)) for j in cand])
            for j in cand[np.argsort(-sims)]:
                if self._gates(x, j):
                    chosen = j
                    break
        if chosen < 0:
            free = np.flatnonzero(~self.committed)
            if free.size == 0:
                self.dropped += 1
                return -1
            j = free[0]
            self.committed[j] = True
            self.anchor[j] = x; self.mu[j] = x.copy(); self.var[j] = self.var0.copy()
            self.conf[j] = 0.0; self.n[j] = 1; self.res[j] = [x]
            return j
        j = chosen
        d = x - self.mu[j]                                    # deviation from pre-update mean
        self.var[j] = (1 - p['beta']) * self.var[j] + p['beta'] * d * d          # Eq. 3
        if p['weighting'] == 'mean':
            eta = 1.0 / (self.n[j] + 1)
        else:
            eta = (1 - self.conf[j]) * (1 - p['alpha_min']) + p['alpha_min']   # Eq. 2
        self.mu[j] = self.mu[j] + eta * d                                       # Eq. 1
        self.conf[j] = p['lam'] * self.conf[j] + (1 - p['lam']) * np.exp(-self.var[j].mean())  # Eq. 4
        self.n[j] += 1
        self.res[j].append(x)
        if len(self.res[j]) > p['reservoir']:
            self.res[j].pop(1)                                 # keep the anchor, drop the oldest other
        return j

    def ready(self):
        return np.flatnonzero(self.committed & (self.n >= self.p['n_ready']))

    def templates(self, cols):
        T = self.mu[cols]
        return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)          # Eq. 5


def run_stage1(X, var0, seed, **kw):
    m = Stage1(X.shape[1], var0, seed=seed, **kw)
    assign = np.array([m.step(x) for x in X])
    ready = m.ready()
    return assign, ready, m.templates(ready) if ready.size else np.zeros((0, X.shape[1])), m
