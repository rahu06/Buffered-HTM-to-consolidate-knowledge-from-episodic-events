"""Comment 1.17 follow-up: improving compositional generalization of the sequence generator.
Variants (all with the entity placeholder): base LSTM; + random-pair augmentation; + attention; + both.
Augmentation: each epoch also trains on random ordered activity pairs (a != b) for training entities, rendered with the
training templates. Pairs that are test targets (ordered (a, b) of any test sample, as presented at test time) are
excluded, so no test pair is ever seen in training."""
import json, random, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, './stage2')
import seqgen as S
from seqgen import ACT, CHAINS, ENT, fact

ACTS = sorted(ACT)


def target(e, a, b, ph=True):
    return f'after {ACT[a][0]} , {"<E>" if ph else e} {ACT[b][1]}'.split()


class S2SAttn(nn.Module):
    """Encoder-decoder LSTM; with attn=True the decoder attends (dot-product) over the two encoder states."""
    def __init__(self, V, attn, H=128, E=64):
        super().__init__()
        self.attn = attn
        self.enc = nn.LSTM(100, H, batch_first=True); self.emb = nn.Embedding(V, E)
        self.cell = nn.LSTMCell(E + (H if attn else 0), H); self.out = nn.Linear(H * (2 if attn else 1), V)

    def step(self, tok, hc, enc, ctx):
        inp = self.emb(tok)
        if self.attn: inp = torch.cat([inp, ctx], -1)
        h, c = self.cell(inp, hc)
        if self.attn:
            w = torch.softmax((enc @ h.unsqueeze(-1)).squeeze(-1), -1)
            ctx = (w.unsqueeze(-1) * enc).sum(1)
            return self.out(torch.cat([h, ctx], -1)), (h, c), ctx
        return self.out(h), (h, c), ctx

    def init(self, x):
        enc, (h, c) = self.enc(x)
        return enc, (h[0], c[0]), torch.zeros_like(h[0])

    def forward(self, x, y_in):
        enc, hc, ctx = self.init(x); outs = []
        for t in range(y_in.shape[1]):
            o, hc, ctx = self.step(y_in[:, t], hc, enc, ctx); outs.append(o)
        return torch.stack(outs, 1)

    def greedy(self, x, bos, L=12):
        enc, hc, ctx = self.init(x); tok = torch.full((x.shape[0],), bos); outs = []
        for _ in range(L):
            o, hc, ctx = self.step(tok, hc, enc, ctx); tok = o.argmax(-1); outs.append(tok)
        return torch.stack(outs, 1)


def run(cond, seed, aug, attn, epochs=1000):
    rng = random.Random(seed); torch.manual_seed(seed)
    combos = [(e, c) for e in ENT for c in range(20)]
    data = rng.sample(combos, 160)
    if cond in ('combinations', 'templates', 'reversed order'):
        rng.shuffle(data); tr, te = data[:120], data[120:]
    elif cond == 'entities':
        test_e = set(rng.sample(ENT, 3)); tr = [d for d in data if d[0] not in test_e]; te = [d for d in data if d[0] in test_e]
    elif cond == 'chains':
        test_c = set(rng.sample(range(20), 5)); tr = [d for d in data if d[1] not in test_c]; te = [d for d in data if d[1] in test_c]
    test_pairs = {(CHAINS[c][1], CHAINS[c][0]) if cond == 'reversed order' else CHAINS[c] for _, c in te}
    if cond == 'chains':
        test_pairs |= {CHAINS[c] for c in test_c}
    train_ents = sorted({e for e, _ in tr})
    aug_pairs = [(a, b) for a in ACTS for b in ACTS if a != b and (a, b) not in test_pairs]
    vocab = ['<pad>', '<bos>', '<eos>'] + sorted({w for a in ACTS for p in ACT[a] for w in p.split()} | {'after', ',', '<E>'})
    ix = {w: i for i, w in enumerate(vocab)}
    model = S2SAttn(len(vocab), attn); opt = torch.optim.Adam(model.parameters(), 1e-3); lossf = nn.CrossEntropyLoss(ignore_index=0)
    for ep in range(epochs):
        items = [(e, *CHAINS[c]) for e, c in tr]
        if aug:
            items += [(rng.choice(train_ents), *rng.choice(aug_pairs)) for _ in range(len(tr))]
        X, Yi, Yo = [], [], []
        for e, a, b in items:
            ts = rng.randrange(2)
            X.append([fact(e, a, ts, rng), fact(e, b, ts, rng)]); t = [ix[w] for w in target(e, a, b)]
            Yi.append([1] + t + [0] * (11 - len(t))); Yo.append(t + [2] + [0] * (11 - len(t)))
        X = torch.tensor(np.array(X)); Yi = torch.tensor(Yi); Yo = torch.tensor(Yo)
        opt.zero_grad(); l = lossf(model(X, Yi).reshape(-1, len(vocab)), Yo.reshape(-1)); l.backward(); opt.step()
    model.eval(); res = []; ex = []
    with torch.no_grad():
        for e, c in te:
            a, b = CHAINS[c]; ts = 2 if cond == 'templates' else rng.randrange(2)
            if cond == 'reversed order':
                a, b = b, a
            x = torch.tensor(np.array([[fact(e, a, ts, rng), fact(e, b, ts, rng)]]))
            out = []
            for t in model.greedy(x, 1)[0].tolist():
                if t == 2: break
                out.append(vocab[t])
            out = [e if w == '<E>' else w for w in out]
            ref = target(e, a, b, False)
            learned = any(out == target(e, *CHAINS[k], False) for _, k in tr)
            ov = sum(min(out.count(w), ref.count(w)) for w in set(ref)) / len(ref)
            res.append(dict(exact=out == ref, overlap=ov, falls_back_to_learned=learned and out != ref))
            if len(ex) < 4: ex.append((' '.join(out), ' '.join(ref)))
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]} | dict(cond=cond, seed=seed, aug=aug, attn=attn, examples=ex)


def job(j):
    torch.set_num_threads(1)
    return run(*j)


if __name__ == '__main__':
    from multiprocessing import Pool
    CONDS = ['combinations', 'templates', 'entities', 'chains', 'reversed order']
    jobs = [(c, s, aug, attn) for aug, attn in ((False, False), (True, False), (False, True), (True, True)) for c in CONDS for s in range(5)]
    with Pool(2) as p:
        out = p.map(job, jobs)
    json.dump(out, open('./stage2/seqgen2_results.json', 'w'), indent=1)
    import collections
    g = collections.defaultdict(list)
    for r in out: g[(r['aug'], r['attn'], r['cond'])].append(r)
    for k, v in g.items():
        print(f"aug={k[0]!s:5} attn={k[1]!s:5} {k[2]:15s} " + ' '.join(
            f"{m}={np.mean([x[m] for x in v]):.3f}±{np.std([x[m] for x in v]):.3f}" for m in ['exact', 'overlap', 'falls_back_to_learned']))
