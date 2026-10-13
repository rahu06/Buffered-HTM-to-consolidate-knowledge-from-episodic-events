"""Additional online / streaming baselines for Reviewer 1, Comment 1.20.
All take the stream X (n x D, unit-normalized embeddings, in presentation order) and return a cluster label per
episode; READY clusters are formed exactly as for the other baselines (clusters with at least min_size members)."""
import numpy as np


def fuzzy_art(X, rho, lo, hi, alpha=0.001, beta=1.0):
    """Fuzzy ART (Carpenter, Grossberg & Rosen, 1991) with complement coding; one pass, in stream order. Inputs are
    scaled to [0, 1] with per-dimension ranges (lo, hi) estimated on the training embeddings."""
    A = np.clip((X - lo) / (hi - lo + 1e-12), 0, 1)
    I = np.hstack([A, 1 - A]); M = I.shape[1] / 2
    W = np.zeros((0, I.shape[1])); lab = np.empty(len(I), int)
    for n, x in enumerate(I):
        if len(W):
            mn = np.minimum(x, W).sum(1)
            T = mn / (alpha + W.sum(1))
            for j in np.argsort(-T):
                if mn[j] / M >= rho:
                    W[j] = beta * np.minimum(x, W[j]) + (1 - beta) * W[j]; lab[n] = j; break
            else:
                W = np.vstack([W, x]); lab[n] = len(W) - 1
        else:
            W = x[None].copy(); lab[n] = 0
    return lab


def leader_memory(X, theta):
    """Online nearest-prototype memory (sequential leader clustering): join the most similar prototype if its cosine
    similarity exceeds theta and update it as a running mean, otherwise store a new prototype."""
    P, cnt, lab = [], [], np.empty(len(X), int)
    for n, x in enumerate(X):
        if P:
            Pm = np.array(P); s = Pm @ x / (np.linalg.norm(Pm, axis=1) + 1e-12); j = int(np.argmax(s))
            if s[j] > theta:
                cnt[j] += 1; P[j] = P[j] + (x - P[j]) / cnt[j]; lab[n] = j; continue
        P.append(x.copy()); cnt.append(1); lab[n] = len(P) - 1
    return lab


def clustream(X, k, seed=0, q=None, t=2.0, n_init=None):
    """CluStream (Aggarwal et al., 2003): q micro-clusters initialised by k-means on the first n_init episodes, then
    online absorption of each episode into its nearest micro-cluster if within t x its RMS radius (for a singleton, the
    distance to the nearest other micro-cluster), otherwise a new micro-cluster is created and the two closest are
    merged; offline, weighted k-means on the micro-cluster centres with k = number of facts (as for K-Means)."""
    from sklearn.cluster import KMeans
    q = q or min(4 * k, len(X) // 2); n_init = n_init or min(len(X), 2 * q)
    km0 = KMeans(q, n_init=3, random_state=seed).fit(X[:n_init])
    LS = [X[:n_init][km0.labels_ == c].sum(0) for c in range(q)]
    SS = [float((X[:n_init][km0.labels_ == c] ** 2).sum()) for c in range(q)]
    N = [int((km0.labels_ == c).sum()) for c in range(q)]
    keep = [c for c in range(q) if N[c] > 0]; LS = [LS[c] for c in keep]; SS = [SS[c] for c in keep]; N = [N[c] for c in keep]
    for x in X[n_init:]:
        C = np.array(LS) / np.array(N)[:, None]; dist = np.linalg.norm(C - x, axis=1); j = int(np.argmin(dist))
        if N[j] > 1:
            r = np.sqrt(max(SS[j] / N[j] - (C[j] @ C[j]), 1e-12))
        else:
            r = np.min(np.delete(np.linalg.norm(C - C[j], axis=1), j)) / t if len(C) > 1 else 0.0
        if dist[j] <= t * r:
            LS[j] = LS[j] + x; SS[j] += float(x @ x); N[j] += 1; continue
        LS.append(x.copy()); SS.append(float(x @ x)); N.append(1)
        C = np.array(LS) / np.array(N)[:, None]; Dm = np.linalg.norm(C[:, None] - C[None], axis=2); np.fill_diagonal(Dm, np.inf)
        a, b = np.unravel_index(np.argmin(Dm), Dm.shape)
        LS[a] = LS[a] + LS[b]; SS[a] += SS[b]; N[a] += N[b]
        del LS[b], SS[b], N[b]
    C = np.array(LS) / np.array(N)[:, None]
    km = KMeans(min(k, len(C)), n_init=10, random_state=seed).fit(C, sample_weight=np.array(N, float))
    return km.predict(X)


def denstream(X, eps, mu=3.0, beta=0.5, lam=0.001):
    """DenStream (Cao et al., 2006): potential and outlier micro-clusters with fading weight 2^(-lam*dt), absorption
    when the radius stays within eps, promotion of outlier micro-clusters whose weight reaches beta*mu; offline,
    density-connected potential micro-clusters (centres within 2*eps) form clusters. Episodes are labelled by the
    cluster of their nearest potential micro-cluster if within eps of its centre, otherwise noise."""
    mcs = []                      # dict(LS, SS, w, t, p)
    for n, x in enumerate(X):
        for m in mcs:
            f = 2 ** (-lam * (n - m['t'])); m['LS'] *= f; m['SS'] *= f; m['w'] *= f; m['t'] = n
        placed = False
        for group in (True, False):
            cand = [m for m in mcs if m['p'] == group]
            if not cand: continue
            C = np.array([m['LS'] / m['w'] for m in cand]); j = int(np.argmin(np.linalg.norm(C - x, axis=1))); m = cand[j]
            LS, SS, w = m['LS'] + x, m['SS'] + x @ x, m['w'] + 1
            c = LS / w; r = np.sqrt(max(SS / w - c @ c, 0))
            if r <= eps:
                m['LS'], m['SS'], m['w'] = LS, SS, w
                if not m['p'] and m['w'] >= beta * mu: m['p'] = True
                placed = True; break
        if not placed:
            mcs.append(dict(LS=x.copy(), SS=float(x @ x), w=1.0, t=n, p=beta * mu <= 1))
    P = [m for m in mcs if m['p'] and m['w'] >= beta * mu]
    if not P:
        return -np.ones(len(X), int)
    C = np.array([m['LS'] / m['w'] for m in P]); W = np.array([m['w'] for m in P])
    from sklearn.cluster import DBSCAN
    mc_lab = DBSCAN(eps=2 * eps, min_samples=1).fit_predict(C)
    dist = np.linalg.norm(X[:, None] - C[None], axis=2); j = dist.argmin(1)
    return np.where(dist[np.arange(len(X)), j] <= 2 * eps, mc_lab[j], -1)
