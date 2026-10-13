"""Column-attention generator (CAG) vs revised LSTM and transformer.
Every input is a set of active memory columns. Each column j is given as its template T_j (100-d fact embedding),
its confidence conf_j and the slot it occupies in the validated sequence (1, 2, 3) from Stage 2. Several columns can
compete for one slot (as in buffered Temporal Memory, where competing predictions coexist); the correct one is the
column of the true fact, which has higher confidence.
CAG: no recurrent encoder; at each output word the decoder (LSTM cell) attends with 4 heads over the column set, with
keys/values from [T_j; slot embedding] and an additive confidence prior alpha*log(conf_j) on the attention scores.
Baselines receive the same columns as an ordered token sequence with [T_j; conf_j; slot one-hot] features.
All methods use the entity slot and random-sequence training (test sequences excluded)."""
import json, math, random, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, './stage2')
from seqgen import ACT, CHAINS, ENT, fact

ACTS = sorted(ACT)
TRI = [(a, b, c) for a, b in CHAINS for b2, c in CHAINS if b == b2 and c != a][:20]
D_IN = 100 + 1 + 3
MAXCOL = 9
L_OUT = 16


def target(e, seq, ph=True):
    E = '<E>' if ph else e
    s = f'after {ACT[seq[0]][0]} , {E} {ACT[seq[1]][1]}'
    if len(seq) == 3:
        s += f' and then {ACT[seq[2]][1]}'
    return s.split()


def columns(e, seq, ts, rng, compete):
    """Return list of (vec, conf, slot, is_true)."""
    cols = []
    for k, a in enumerate(seq):
        if compete:
            cols.append((fact(e, a, ts, rng), rng.uniform(0.6, 1.0), k + 1, True))
            for _ in range(rng.randint(*compete)):
                o = rng.choice([x for x in ACTS if x not in seq])
                cols.append((fact(e, o, ts, rng), rng.uniform(0.1, 0.5), k + 1, False))
        else:
            cols.append((fact(e, a, ts, rng), rng.uniform(0.6, 1.0), k + 1, True))
    rng.shuffle(cols); cols.sort(key=lambda c: c[2])
    return cols


def encode(cols):
    X = np.zeros((MAXCOL, D_IN), np.float32); m = np.zeros(MAXCOL, bool); slot = np.zeros(MAXCOL, np.int64); conf = np.ones(MAXCOL, np.float32)
    for i, (v, c, s, _) in enumerate(cols):
        X[i, :100] = v; X[i, 100] = c; X[i, 100 + s] = 1; m[i] = True; slot[i] = s; conf[i] = c
    return X, m, slot, conf


class CAG(nn.Module):
    def __init__(self, V, H=128, E=64, heads=4):
        super().__init__()
        self.h, self.dk = heads, H // heads
        self.slot = nn.Embedding(4, 32)
        self.k = nn.Linear(132, H); self.v = nn.Linear(132, H); self.q = nn.Linear(H, H)
        self.alpha = nn.Parameter(torch.ones(heads))
        self.init = nn.Linear(H, 2 * H)
        self.emb = nn.Embedding(V, E); self.cell = nn.LSTMCell(E + H, H); self.out = nn.Linear(2 * H, V)

    def memory(self, X, m, slot, conf):
        f = torch.cat([X[..., :100], self.slot(slot)], -1)
        K = self.k(f).view(*f.shape[:2], self.h, self.dk); Vv = self.v(f).view(*f.shape[:2], self.h, self.dk)
        prior = torch.log(conf.clamp_min(1e-3))[:, :, None] * self.alpha            # B x N x heads
        w0 = torch.where(m, conf, torch.zeros_like(conf)); w0 = w0 / w0.sum(1, keepdim=True)
        h0, c0 = self.init((w0[..., None] * self.v(f)).sum(1)).tanh().chunk(2, -1)
        return K, Vv, prior, (h0, c0)

    def attend(self, h, K, Vv, prior, m):
        q = self.q(h).view(h.shape[0], self.h, self.dk)
        s = torch.einsum('bhd,bnhd->bnh', q, K) / math.sqrt(self.dk) + prior
        s = s.masked_fill(~m[..., None], -1e9); a = torch.softmax(s, 1)
        return torch.einsum('bnh,bnhd->bhd', a, Vv).reshape(h.shape[0], -1), a

    def run(self, X, m, slot, conf, y_in=None, bos=1, L=L_OUT):
        K, Vv, prior, (h, c) = self.memory(X, m, slot, conf)
        ctx = torch.zeros_like(h); outs = []; atts = []
        tok = torch.full((X.shape[0],), bos)
        for t in range(L if y_in is None else y_in.shape[1]):
            if y_in is not None: tok = y_in[:, t]
            h, c = self.cell(torch.cat([self.emb(tok), ctx], -1), (h, c))
            ctx, a = self.attend(h, K, Vv, prior, m)
            o = self.out(torch.cat([h, ctx], -1)); outs.append(o); atts.append(a.mean(-1))
            if y_in is None: tok = o.argmax(-1)
        return torch.stack(outs, 1), torch.stack(atts, 1)

    def forward(self, X, m, slot, conf, y_in):
        return self.run(X, m, slot, conf, y_in)[0]

    def greedy(self, X, m, slot, conf):
        o, a = self.run(X, m, slot, conf); return o.argmax(-1), a


class LSTMAttn(nn.Module):
    def __init__(self, V, H=128, E=64):
        super().__init__()
        self.enc = nn.LSTM(D_IN, H, batch_first=True); self.emb = nn.Embedding(V, E)
        self.cell = nn.LSTMCell(E + H, H); self.out = nn.Linear(2 * H, V)

    def run(self, X, m, slot, conf, y_in=None, bos=1, L=L_OUT):
        lens = m.sum(1)
        packed = nn.utils.rnn.pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
        enc, (h, c) = self.enc(packed); enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=X.shape[1])
        h, c = h[0], c[0]; ctx = torch.zeros_like(h); outs = []; tok = torch.full((X.shape[0],), bos)
        for t in range(L if y_in is None else y_in.shape[1]):
            if y_in is not None: tok = y_in[:, t]
            h, c = self.cell(torch.cat([self.emb(tok), ctx], -1), (h, c))
            s = (enc @ h[..., None]).squeeze(-1).masked_fill(~m, -1e9); a = torch.softmax(s, 1)
            ctx = (a[..., None] * enc).sum(1); o = self.out(torch.cat([h, ctx], -1)); outs.append(o)
            if y_in is None: tok = o.argmax(-1)
        return torch.stack(outs, 1)

    def forward(self, X, m, slot, conf, y_in):
        return self.run(X, m, slot, conf, y_in)

    def greedy(self, X, m, slot, conf):
        return self.run(X, m, slot, conf).argmax(-1), None


class TF(nn.Module):
    def __init__(self, V, d=64):
        super().__init__()
        self.inp = nn.Linear(D_IN, d); self.emb = nn.Embedding(V, d); self.pos = nn.Parameter(torch.randn(L_OUT + 1, d) * 0.02)
        self.tf = nn.Transformer(d, 4, 2, 2, 128, 0.1, batch_first=True); self.out = nn.Linear(d, V)

    def forward(self, X, m, slot, conf, y_in):
        T = y_in.shape[1]
        h = self.tf(self.inp(X), self.emb(y_in) + self.pos[:T], tgt_mask=self.tf.generate_square_subsequent_mask(T),
                    src_key_padding_mask=~m, memory_key_padding_mask=~m)
        return self.out(h)

    def greedy(self, X, m, slot, conf):
        y = torch.full((X.shape[0], 1), 1)
        for _ in range(L_OUT):
            y = torch.cat([y, self.forward(X, m, slot, conf, y)[:, -1].argmax(-1, keepdim=True)], 1)
        return y[:, 1:], None


def setup(cond, rng):
    if cond == 'longer sequences (3 facts)':
        combos = [(e, k) for e in ENT for k in range(len(TRI))]; data = rng.sample(combos, 160)
        test_k = set(rng.sample(range(len(TRI)), 5))
        tr = [(e, TRI[k]) for e, k in data if k not in test_k]; te = [(e, TRI[k]) for e, k in data if k in test_k]
        return tr, te, {TRI[k] for k in test_k}
    combos = [(e, c) for e in ENT for c in range(20)]; data = rng.sample(combos, 160)
    if cond in ('combinations', 'templates', 'reversed order', 'competing columns', 'more competitors at test'):
        rng.shuffle(data); tr, te = data[:120], data[120:]
    elif cond == 'entities':
        te_e = set(rng.sample(ENT, 3)); tr = [d for d in data if d[0] not in te_e]; te = [d for d in data if d[0] in te_e]
    elif cond == 'chains':
        te_c = set(rng.sample(range(20), 5)); tr = [d for d in data if d[1] not in te_c]; te = [d for d in data if d[1] in te_c]
        tr = [(e, CHAINS[c]) for e, c in tr]; te = [(e, CHAINS[c]) for e, c in te]
        return tr, te, {CHAINS[c] for c in te_c} | {s for _, s in te}
    tr = [(e, CHAINS[c]) for e, c in tr]
    te = [(e, CHAINS[c][::-1] if cond == 'reversed order' else CHAINS[c]) for e, c in te]
    return tr, te, {s for _, s in te}


def run(method, cond, seed, epochs=1000):
    rng = random.Random(seed); torch.manual_seed(seed)
    tr, te, test_seqs = setup(cond, rng)
    compete = (1, 2) if cond == 'competing columns' else (1, 1) if cond == 'more competitors at test' else None
    compete_test = (3, 3) if cond == 'more competitors at test' else compete
    ents = sorted({e for e, _ in tr})
    vocab = ['<pad>', '<bos>', '<eos>'] + sorted({w for a in ACTS for p in ACT[a] for w in p.split()} | {'after', ',', '<E>', 'and', 'then'})
    ix = {w: i for i, w in enumerate(vocab)}
    model = {'column attention (CAG)': CAG, 'revised LSTM': LSTMAttn, 'transformer': TF}[method](len(vocab))
    opt = torch.optim.Adam(model.parameters(), 1e-3); lossf = nn.CrossEntropyLoss(ignore_index=0)
    lens = sorted({len(s) for _, s in tr})

    def rand_seq():
        while True:
            s = tuple(rng.sample(ACTS, rng.choice(lens)))
            if s not in test_seqs: return s

    def batch(items, tsets, compete=compete):
        Xs, Ms, Ss, Cs, Yi, Yo = [], [], [], [], [], []
        for e, s in items:
            X, m, sl, cf = encode(columns(e, s, rng.choice(tsets), rng, compete))
            t = [ix[w] for w in target(e, s)]
            Xs.append(X); Ms.append(m); Ss.append(sl); Cs.append(cf)
            Yi.append([1] + t + [0] * (L_OUT - 1 - len(t))); Yo.append(t + [2] + [0] * (L_OUT - 1 - len(t)))
        T = lambda a: torch.tensor(np.array(a))
        return T(Xs), T(Ms), T(Ss), T(Cs), T(Yi), T(Yo)

    for ep in range(epochs):
        model.train()
        items = list(tr) + [(rng.choice(ents), rand_seq()) for _ in range(len(tr))]
        X, M, S, C, Yi, Yo = batch(items, [0, 1])
        opt.zero_grad(); l = lossf(model(X, M, S, C, Yi).reshape(-1, len(vocab)), Yo.reshape(-1)); l.backward(); opt.step()
    model.eval(); res = []
    with torch.no_grad():
        Xs, Ms, Ss, Cs, truth, colsets = [], [], [], [], [], []
        for e, s in te:
            cols = columns(e, s, 2 if cond == 'templates' else rng.randrange(2), rng, compete_test)
            X, m, sl, cf = encode(cols); Xs.append(X); Ms.append(m); Ss.append(sl); Cs.append(cf); colsets.append(cols)
        T = lambda a: torch.tensor(np.array(a))
        pred, att = model.greedy(T(Xs), T(Ms), T(Ss), T(Cs))
        for i, (e, s) in enumerate(te):
            out = []
            for t in pred[i].tolist():
                if t == 2: break
                out.append(vocab[t])
            out = [e if w == '<E>' else w for w in out]; ref = target(e, s, False)
            r = dict(exact=out == ref, overlap=sum(min(out.count(w), ref.count(w)) for w in set(ref)) / len(ref))
            if att is not None:
                n = int(Ms[i].sum()); true_idx = [j for j, c in enumerate(colsets[i]) if c[3]]
                r['attention_on_true'] = float(att[i, :len(ref), true_idx].sum(-1).mean())
            res.append(r)
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    return out | dict(method=method, cond=cond, seed=seed, n_test=len(te))


def job(j):
    torch.set_num_threads(1)
    return run(*j)


if __name__ == '__main__':
    from multiprocessing import Pool
    M = ['column attention (CAG)', 'revised LSTM', 'transformer']
    C = ['combinations', 'templates', 'entities', 'chains', 'reversed order', 'longer sequences (3 facts)', 'competing columns', 'more competitors at test']
    jobs = [(m, c, s) for c in C for m in M for s in range(5)]
    with Pool(2) as p:
        out = p.map(job, jobs)
    json.dump(out, open('./stage2/seqgen4_results.json', 'w'), indent=1)
    import collections
    g = collections.defaultdict(list)
    for r in out: g[(r['cond'], r['method'])].append(r)
    for k, v in g.items():
        print(f"{k[0]:27s} {k[1]:24s} exact={np.mean([x['exact'] for x in v]):.3f}±{np.std([x['exact'] for x in v]):.3f} "
              f"overlap={np.mean([x['overlap'] for x in v]):.3f}" + (f" att_true={np.mean([x['attention_on_true'] for x in v]):.2f}" if 'attention_on_true' in v[0] else ''))
