"""Comment 1.19: slot-level metrics, uncertainty for all metrics, and stronger baselines for sequential generation.
Methods (same data, splits and inputs): original LSTM generator (no entity slot); revised LSTM generator (entity slot,
random-pair training, attention); transformer encoder-decoder (entity slot, random-pair training); relation-aware
template model (per-fact activity classifier + template "after <A>, <E> <B>", entity from the slot).
Metrics per test sentence: exact match; entity, antecedent and consequent accuracy (parsed slots); order accuracy
(both correct activities in the correct positions, among outputs that name both correct activities); relation
accuracy (output is a well-formed "after X, E Y" statement with two known activities); word overlap (fraction of
reference tokens present in the output, clipped by count); embedding similarity (cosine of mean GloVe-100 vectors)."""
import json, math, random, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, './stage2')
from seqgen import ACT, CHAINS, ENT, fact, G
from seqgen2 import S2SAttn

ACTS = sorted(ACT)
GER = {tuple(ACT[a][0].split()): a for a in ACTS}
V3 = {tuple(ACT[a][1].split()): a for a in ACTS}


def target(e, a, b, ph):
    return f'after {ACT[a][0]} , {"<E>" if ph else e} {ACT[b][1]}'.split()


def parse(out):
    """Return (antecedent, entity, consequent, well_formed)."""
    if len(out) < 4 or out[0] != 'after' or ',' not in out:
        return None, None, None, False
    k = out.index(',')
    ante = GER.get(tuple(out[1:k])); ent = out[k + 1] if len(out) > k + 1 else None
    cons = V3.get(tuple(out[k + 2:]))
    return ante, ent, cons, ante is not None and cons is not None and ent is not None


def semb(toks):
    v = [G[w] for w in toks if w in G]
    if not v: return np.zeros(100)
    x = np.mean(v, 0); return x / (np.linalg.norm(x) + 1e-9)


class TransformerS2S(nn.Module):
    def __init__(self, V, d=64, L=12):
        super().__init__()
        self.inp = nn.Linear(100, d); self.src_pos = nn.Parameter(torch.randn(2, d) * 0.02)
        self.emb = nn.Embedding(V, d); self.pos = nn.Parameter(torch.randn(L + 1, d) * 0.02)
        self.tf = nn.Transformer(d, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.out = nn.Linear(d, V)

    def forward(self, x, y_in):
        T = y_in.shape[1]; mask = self.tf.generate_square_subsequent_mask(T)
        h = self.tf(self.inp(x) + self.src_pos, self.emb(y_in) + self.pos[:T], tgt_mask=mask)
        return self.out(h)

    def greedy(self, x, bos, L=12):
        y = torch.full((x.shape[0], 1), bos)
        for _ in range(L):
            nxt = self.forward(x, y)[:, -1].argmax(-1, keepdim=True); y = torch.cat([y, nxt], 1)
        return y[:, 1:]


def split(cond, rng):
    combos = [(e, c) for e in ENT for c in range(20)]
    data = rng.sample(combos, 160); test_c = set()
    if cond in ('combinations', 'templates', 'reversed order'):
        rng.shuffle(data); tr, te = data[:120], data[120:]
    elif cond == 'entities':
        test_e = set(rng.sample(ENT, 3)); tr = [d for d in data if d[0] not in test_e]; te = [d for d in data if d[0] in test_e]
    else:
        test_c = set(rng.sample(range(20), 5)); tr = [d for d in data if d[1] not in test_c]; te = [d for d in data if d[1] in test_c]
    return tr, te, test_c


def run(method, cond, seed, epochs=1000):
    rng = random.Random(seed); torch.manual_seed(seed)
    tr, te, test_c = split(cond, rng)
    ph = method != 'original LSTM'
    test_pairs = {(CHAINS[c][1], CHAINS[c][0]) if cond == 'reversed order' else CHAINS[c] for _, c in te} | {CHAINS[c] for c in test_c}
    train_ents = sorted({e for e, _ in tr})
    aug_pairs = [(a, b) for a in ACTS for b in ACTS if a != b and (a, b) not in test_pairs]
    if method == 'relation-aware template':
        clf = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, len(ACTS))); opt = torch.optim.Adam(clf.parameters(), 1e-2)
        for ep in range(300):
            X, Y = [], []
            for e, c in tr:
                for a in CHAINS[c]:
                    X.append(fact(e, a, rng.randrange(2), rng)); Y.append(ACTS.index(a))
            l = nn.functional.cross_entropy(clf(torch.tensor(np.array(X))), torch.tensor(Y)); opt.zero_grad(); l.backward(); opt.step()
        gen = lambda xa, xb: target('<E>', ACTS[clf(torch.tensor(xa)[None]).argmax()], ACTS[clf(torch.tensor(xb)[None]).argmax()], True)
    else:
        if ph:
            vocab = ['<pad>', '<bos>', '<eos>'] + sorted({w for a in ACTS for p in ACT[a] for w in p.split()} | {'after', ',', '<E>'})
        else:
            vocab = ['<pad>', '<bos>', '<eos>'] + sorted({w for e, c in tr for w in target(e, *CHAINS[c], False)})
        ix = {w: i for i, w in enumerate(vocab)}
        model = TransformerS2S(len(vocab)) if method == 'transformer' else S2SAttn(len(vocab), attn=(method == 'revised LSTM'))
        opt = torch.optim.Adam(model.parameters(), 1e-3); lossf = nn.CrossEntropyLoss(ignore_index=0)
        for ep in range(epochs):
            model.train()
            items = [(e, *CHAINS[c]) for e, c in tr]
            if method in ('revised LSTM', 'transformer'):
                items += [(rng.choice(train_ents), *rng.choice(aug_pairs)) for _ in range(len(tr))]
            X, Yi, Yo = [], [], []
            for e, a, b in items:
                ts = rng.randrange(2)
                X.append([fact(e, a, ts, rng), fact(e, b, ts, rng)]); t = [ix[w] for w in target(e, a, b, ph)]
                Yi.append([1] + t + [0] * (11 - len(t))); Yo.append(t + [2] + [0] * (11 - len(t)))
            X = torch.tensor(np.array(X)); Yi = torch.tensor(Yi); Yo = torch.tensor(Yo)
            opt.zero_grad(); l = lossf(model(X, Yi).reshape(-1, len(vocab)), Yo.reshape(-1)); l.backward(); opt.step()
        model.eval()

        def gen(xa, xb):
            out = []
            for t in model.greedy(torch.tensor(np.array([[xa, xb]])), 1)[0].tolist():
                if t == 2: break
                out.append(vocab[t])
            return out
    res = []
    with torch.no_grad():
        for e, c in te:
            a, b = CHAINS[c]; ts = 2 if cond == 'templates' else rng.randrange(2)
            if cond == 'reversed order':
                a, b = b, a
            out = gen(fact(e, a, ts, rng), fact(e, b, ts, rng))
            out = [e if w == '<E>' else w for w in out]
            ref = target(e, a, b, False)
            pa, pe, pc, wf = parse(out)
            both = {pa, pc} == {a, b}
            res.append(dict(exact=out == ref, entity=pe == e, antecedent=pa == a, consequent=pc == b,
                            order=(pa == a and pc == b) if both else float('nan'), relation=wf,
                            overlap=sum(min(out.count(w), ref.count(w)) for w in set(ref)) / len(ref),
                            similarity=float(semb(out) @ semb(ref))))
    summ = {k: float(np.nanmean([r[k] for r in res])) if not all(math.isnan(r[k]) if isinstance(r[k], float) else False for r in res) else float('nan') for k in res[0]}
    summ['order_n'] = int(sum(not (isinstance(r['order'], float) and math.isnan(r['order'])) for r in res))
    return summ | dict(method=method, cond=cond, seed=seed, n_test=len(te))


def job(j):
    torch.set_num_threads(1)
    return run(*j)


if __name__ == '__main__':
    from multiprocessing import Pool
    M = ['original LSTM', 'revised LSTM', 'transformer', 'relation-aware template']
    C = ['combinations', 'templates', 'entities', 'chains', 'reversed order']
    jobs = [(m, c, s) for m in M for c in C for s in range(5)]
    with Pool(2) as p:
        out = p.map(job, jobs)
    json.dump(out, open('./stage2/seqgen3_results.json', 'w'), indent=1)
    import collections
    g = collections.defaultdict(list)
    for r in out: g[(r['method'], r['cond'])].append(r)
    K = ['exact', 'entity', 'antecedent', 'consequent', 'order', 'relation', 'overlap', 'similarity']
    for k_, v in g.items():
        print(f"{k_[0]:24s} {k_[1]:15s} " + ' '.join(f"{m}={np.nanmean([x[m] for x in v]):.3f}±{np.nanstd([x[m] for x in v]):.3f}" for m in K)
              + f" order_n={sum(x['order_n'] for x in v)}")
