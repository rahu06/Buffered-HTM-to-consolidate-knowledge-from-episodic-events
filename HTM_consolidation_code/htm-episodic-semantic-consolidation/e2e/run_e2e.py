"""Reviewer 1, Comment 1.2: end-to-end evaluation, Stage 1 -> Stage 2 -> sentence generator.

Each of 12 entities follows two daily routines (antecedent -> consequent activity chains from the Stage 2 vocabulary)
over 10 days, plus occasional activities and unrelated (negative-control) episodes. Every activity occurrence is an
episodic sentence with varied wording and items. The episodes of all entities are interleaved into one stream.

Stage 1 (final configuration: similarity routing, running mean, column merging, no extra gates; vigilance tau selected
on a separate validation stream by B-cubed F1) consolidates the stream online; READY columns are decoded against the
closed catalogue of entity-activity facts. Stage 2 receives, for each entity and day, the sequence of READY columns
of that entity's episodes and learns transitions with the column-level detector of Eq. 8 (W <- W + eta (1 - W) on a
co-occurrence, W <- gamma W otherwise; a rule is accepted when support >= s_min and W >= w_min). Standard: only the
immediate successor counts. Buffer-enhanced: a successor counts while the antecedent's prediction survives in the
buffer (L0 = 3, d = 0.6, theta_L = 0.5 -> up to three intervening events). Each accepted rule is forwarded, with the
decoded facts of its two columns, to the transformer generator (entity slot, trained on random activity pairs that
exclude all routine chains). The same Stage 2 and generator are also run on oracle columns (true facts).
Insertion noise: with probability p, 1-3 unrelated activities are inserted between antecedent and consequent."""
import gzip, json, random, re, sys, time
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
from multiprocessing import Pool
sys.path.insert(0, '.'); sys.path.insert(0, './stage2')
import htm_v2
from htm_v2 import GloveEncoder, run_v2
import run_clinc as R
import seqgen4 as SG
from seqgen import ACT, CHAINS, ENT, fact

ACTS = sorted(ACT)
EP = {  # episodic templates per activity: {e} entity, {x} item, {w} when
    'breakfast': (['{e} ate {x} for breakfast {w}', '{e} had {x} in the morning {w}', 'for breakfast {e} made {x} {w}'], ['idli', 'poha', 'toast', 'eggs', 'cereal']),
    'cricket': (['{e} played cricket with {x} {w}', '{e} batted in a cricket match with {x} {w}', '{e} joined {x} for a game of cricket {w}'], ['friends', 'neighbours', 'colleagues', 'cousins']),
    'yoga': (['{e} did {x} yoga {w}', '{e} practised {x} yoga at home {w}', '{e} attended a {x} yoga session {w}'], ['morning', 'gentle', 'hatha', 'breathing']),
    'shower': (['{e} took a {x} shower {w}', '{e} had a quick {x} shower {w}', '{e} washed under a {x} shower {w}'], ['hot', 'cold', 'warm', 'long']),
    'meeting': (['{e} attended a meeting with the {x} {w}', '{e} joined the {x} meeting {w}', '{e} met the {x} in a meeting room {w}'], ['team', 'manager', 'clients', 'board']),
    'lunch': (['{e} ate {x} for lunch {w}', '{e} had {x} at lunchtime {w}', 'at lunch {e} ordered {x} {w}'], ['rice', 'dal', 'roti', 'curry', 'salad']),
    'work': (['{e} started work at the {x} {w}', '{e} began working in the {x} {w}', '{e} logged in at the {x} {w}'], ['office', 'desk', 'factory', 'studio']),
    'walk': (['{e} went for a walk in the {x} {w}', '{e} walked around the {x} {w}', '{e} took a stroll through the {x} {w}'], ['park', 'garden', 'colony', 'market']),
    'gym': (['{e} went to the gym and lifted {x} {w}', '{e} trained at the gym on the {x} {w}', '{e} visited the gym for {x} {w}'], ['weights', 'treadmill', 'cardio', 'machines']),
    'dinner': (['{e} cooked {x} for dinner {w}', '{e} prepared {x} in the kitchen for dinner {w}', 'for dinner {e} made {x} {w}'], ['pasta', 'biryani', 'soup', 'paneer']),
    'reading': (['{e} read a {x} book {w}', '{e} spent time reading a {x} novel {w}', '{e} opened a {x} book and read {w}'], ['mystery', 'history', 'science', 'poetry']),
    'music': (['{e} listened to {x} music {w}', '{e} played some {x} songs {w}', '{e} put on {x} music {w}'], ['classical', 'jazz', 'folk', 'pop']),
    'shopping': (['{e} went shopping for {x} {w}', '{e} bought {x} at the mall {w}', '{e} visited the shops to buy {x} {w}'], ['clothes', 'groceries', 'shoes', 'gifts']),
    'tea': (['{e} drank a cup of {x} tea {w}', '{e} made {x} tea {w}', '{e} sipped {x} tea {w}'], ['masala', 'green', 'ginger', 'black'])}
WHEN = ['today', 'again', 'as usual', 'on monday', 'on tuesday', 'on friday', 'on sunday', '']
NEG = ['{e} saw a {x} car on the road', '{e} heard that the {x} would be late', '{e} noticed a {x} poster on the wall',
       '{e} received a {x} message from a stranger', '{e} found a {x} umbrella near the gate', '{e} dreamt about a {x} river',
       '{e} lost a {x} pen somewhere', '{e} watched a {x} bird outside the window', '{e} forgot the {x} password once',
       '{e} spotted a {x} balloon in the sky']
NEG_X = ['red', 'blue', 'train', 'bus', 'strange', 'funny', 'new', 'old', 'broken', 'tiny', 'yellow', 'noisy']
DAYS, P_NEG = 10, 0.12
TRAIN_ENT = ['sanjay', 'pooja', 'ravi', 'deepa', 'manoj', 'nisha', 'ajay', 'geeta', 'vijay', 'lata', 'suresh', 'asha']
ETA, GAMMA, S_MIN, W_MIN = 0.3, 0.9, 7, 0.6
fact_text = lambda e, a: f'{e} {ACT[a][1]}'

# GloVe for this vocabulary
need = {w for a in EP for t in EP[a][0] for w in re.findall(r'[a-z]+', t)} | {x for a in EP for x in EP[a][1]} | set(ENT) | set(TRAIN_ENT) | \
       {w for t in WHEN for w in t.split()} | {w for t in NEG for w in re.findall(r'[a-z]+', t)} | set(NEG_X) | \
       {w for a in ACT for p in ACT[a] for w in p.split()}
G = {}
with gzip.open('./glove100.gz', 'rt', encoding='utf8') as f:
    for line in f:
        w, rest = line.split(' ', 1)
        if w in need: G[w] = np.array(rest.split(), dtype=np.float32)
htm_v2._G = G


def make_stream(seed, p_ins, ents=ENT):
    ENT = ents
    rng = random.Random(seed)
    plan = {}
    for e in ENT:
        while True:
            c = rng.sample(CHAINS, 2)
            if len({c[0][0], c[0][1], c[1][0], c[1][1]}) == 4: break
        others = [a for a in ACTS if a not in {c[0][0], c[0][1], c[1][0], c[1][1]}]
        plan[e] = dict(chains=c, occasional=rng.sample(others, 3), fill=others)
    episodes = []
    for day in range(DAYS):
        tl = {}
        for e in ENT:
            p = plan[e]; seq = []
            chains = p['chains'][:]; rng.shuffle(chains)
            for k, (a, b) in enumerate(chains):
                seq.append(a)
                if rng.random() < p_ins:
                    seq += rng.sample(p['fill'], rng.randint(1, 3))
                seq.append(b)
                if k == 0: seq += rng.sample(p['occasional'], rng.randint(1, 2))
            evs = []
            for a in seq:
                t, items = EP[a]
                evs.append((rng.choice(t).format(e=e, x=rng.choice(items), w=rng.choice(WHEN)).strip(), e, fact_text(e, a), a))
                if rng.random() < P_NEG:
                    evs.append((rng.choice(NEG).format(e=e, x=rng.choice(NEG_X)), e, 'oos', None))
            tl[e] = evs
        ptr = {e: 0 for e in ENT}
        while any(ptr[e] < len(tl[e]) for e in ENT):     # random interleaving that keeps each entity's order
            e = rng.choice([e for e in ENT if ptr[e] < len(tl[e])])
            episodes.append(tl[e][ptr[e]] + (day,)); ptr[e] += 1
    return episodes, plan


def detect(day_seqs, buffered, s_min=S_MIN, w_min=W_MIN, weighted=False, c_min=0.0):
    """Eq. 8 detector. weighted: a successor at distance k (k-1 intervening events) gets credit L0*d^(k-1)/L0;
    c_min: minimum share of the days with A on which B followed A (consistency)."""
    W, S, NA = {}, {}, {}
    win = 4 if buffered else 1
    for seq in day_seqs:
        seen = {}
        for i, a in enumerate(seq):
            for k, b in enumerate(seq[i + 1:i + 1 + win], 1):
                if b != a:
                    cr = 0.6 ** (k - 1) if weighted else 1.0
                    seen[(a, b)] = max(seen.get((a, b), 0.0), cr)
        ants = set(seq)
        for a in ants: NA[a] = NA.get(a, 0) + 1
        for key in list(W):
            if key[0] in ants and key not in seen: W[key] *= GAMMA
        for key, cr in seen.items():
            W[key] = W.get(key, 0.0) + ETA * cr * (1 - W.get(key, 0.0)); S[key] = S.get(key, 0) + 1
    return {k for k in W if S[k] >= s_min and W[k] >= w_min and S[k] / NA[k[0]] >= c_min}


def train_generator(seed, epochs=1000):
    rng = random.Random(seed); torch.manual_seed(seed)
    vocab = ['<pad>', '<bos>', '<eos>'] + sorted({w for a in ACTS for p in ACT[a] for w in p.split()} | {'after', ',', '<E>', 'and', 'then'})
    ix = {w: i for i, w in enumerate(vocab)}
    model = SG.TF(len(vocab)); opt = torch.optim.Adam(model.parameters(), 1e-3); lossf = nn.CrossEntropyLoss(ignore_index=0)
    excl = set(CHAINS)
    for ep in range(epochs):
        Xs, Ms, Ss, Cs, Yi, Yo = [], [], [], [], [], []
        for _ in range(240):
            while True:
                s = tuple(rng.sample(ACTS, 2))
                if s not in excl: break
            e = rng.choice(ENT)
            X, m, sl, cf = SG.encode(SG.columns(e, s, rng.choice([0, 1]), rng, None)); t = [ix[w] for w in SG.target(e, s)]
            Xs.append(X); Ms.append(m); Ss.append(sl); Cs.append(cf)
            Yi.append([1] + t + [0] * (SG.L_OUT - 1 - len(t))); Yo.append(t + [2] + [0] * (SG.L_OUT - 1 - len(t)))
        T = lambda a: torch.tensor(np.array(a))
        opt.zero_grad(); l = lossf(model(T(Xs), T(Ms), T(Ss), T(Cs), T(Yi)).reshape(-1, len(vocab)), T(Yo).reshape(-1)); l.backward(); opt.step()
    model.eval()
    return model, vocab


def generate(model, vocab, items, rng):
    """items: list of (entity, actA, actB) taken from decoded facts -> generated word lists."""
    if not items: return []
    Xs, Ms, Ss, Cs = [], [], [], []
    for e, a, b in items:
        X, m, sl, cf = SG.encode(SG.columns(e, (a, b), 2, rng, None)); Xs.append(X); Ms.append(m); Ss.append(sl); Cs.append(cf)
    T = lambda a: torch.tensor(np.array(a))
    with torch.no_grad():
        pred, _ = model.greedy(T(Xs), T(Ms), T(Ss), T(Cs))
    outs = []
    for i, (e, a, b) in enumerate(items):
        o = []
        for t in pred[i].tolist():
            if t == 2: break
            o.append(vocab[t])
        outs.append([e if w == '<E>' else w for w in o])
    return outs


CATALOGUE = [(e, a) for e in ENT for a in ACTS]


def stage1(texts, ents, enc, var0, tau, seed, ENT=ENT):
    """Entity-indexed memory: each entity has its own column population; returns global column ids."""
    X = enc(texts); a = np.full(len(texts), -1); rdy_all, T_all = [], []
    kw = dict(n_columns=256, n_ready=2, routing='similarity', tau=tau, weighting='mean', use_anchor=False, use_reservoir=False,
              z_thresh=np.inf, merge=True, tau_merge=0.85)
    for k, e in enumerate(ENT):
        idx = np.array([i for i, x in enumerate(ents) if x == e])
        ae, rdy, T, m = run_v2(X[idx], var0, seed + k, **kw)
        a[idx] = np.where(ae >= 0, ae + 1000 * k, -1)
        rdy_all += [r + 1000 * k for r in rdy]; T_all.append(T)
    return a, np.array(rdy_all, dtype=int), np.vstack(T_all)


def find_rules(eps, a, dec, src, buffered, s_min, w_min, weighted=False, c_min=0.0, ENT=ENT):
    """Stage 2 on one stream. Stage-1 source: each episode in a READY column is represented by the decoded fact of its
    column (columns decoding to the same fact are one node); oracle source: the true fact."""
    rules = set()
    for e in ENT:
        days = []
        for d in range(DAYS):
            if src == 'stage1':
                seq = [dec[a[i]] for i, x in enumerate(eps) if x[1] == e and x[4] == d and a[i] in dec]
            else:
                seq = [(e, x[3]) for x in eps if x[1] == e and x[4] == d and x[3] is not None]
            days.append(seq)
        for (ea, aa), (eb, ab) in detect(days, buffered, s_min, w_min, weighted, c_min):
            if aa != ab: rules.add((ea, aa, ab))
    return rules


def job(seed):
    from htm_consolidation import train_contrastive
    t0 = time.time(); results = []
    model, vocab = train_generator(seed)
    for p_ins in (0.0, 0.3, 0.6):
        val_eps, val_plan = make_stream(5000 + seed, p_ins, TRAIN_ENT)       # training entities: fitting and selection
        vt = [x[0] for x in val_eps]; vy = [x[2] for x in val_eps]; ve = [x[1] for x in val_eps]
        eps, plan = make_stream(seed, p_ins)                                    # unseen test entities
        texts = [x[0] for x in eps]; y = [x[2] for x in eps]; te = [x[1] for x in eps]
        vcat_items = [(e, a_) for e in TRAIN_ENT for a_ in ACTS]
        true_rules = {(e, a_, b_) for e in ENT for a_, b_ in plan[e]['chains']}
        val_true = {(e, a_, b_) for e in TRAIN_ENT for a_, b_ in val_plan[e]['chains']}
        for variant in ('contrastive projection', 'fact-anchored projection'):
            enc = GloveEncoder().fit(vt)
            from collections import Counter as _C
            cnt = _C(vy); pos = [i for i, l in enumerate(vy) if l != 'oos' and cnt[l] >= 2]
            tx, ly = [vt[i] for i in pos], [vy[i] for i in pos]
            if variant == 'fact-anchored projection':      # the fact sentence joins its episodes as a positive (training entities only)
                fl = sorted(set(ly)); tx += fl + fl; ly += fl + fl
            enc.proj = train_contrastive(enc.base(tx), ly, batch=min(64, len(set(ly))), seed=seed)
            Xv = enc(vt); labs = np.array(vy)
            var0 = np.mean([Xv[labs == l].var(0) for l in set(vy) if l != 'oos'], axis=0)
            vcat_txt = [fact_text(e, a_) for e, a_ in vcat_items]; vcat = enc(vcat_txt)
            cat_txt = [fact_text(e, a_) for e, a_ in CATALOGUE]; cat = enc(cat_txt)
            best = max([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], key=lambda tau: R.evaluate(vy, *stage1(vt, ve, enc, var0, tau, seed, TRAIN_ENT), vcat, vcat_txt)['BcF1'])
            va, vr, vT = stage1(vt, ve, enc, var0, best, seed, TRAIN_ENT)
            vdec = {c: vcat_items[k] for c, k in zip(vr, np.argmax(vT @ vcat.T, axis=1))} if len(vr) else {}
            a, rdy, T = stage1(texts, te, enc, var0, best, seed)
            ev = R.evaluate(y, a, rdy, T, cat, cat_txt)
            dec = {c: CATALOGUE[k] for c, k in zip(rdy, np.argmax(T @ cat.T, axis=1))} if len(rdy) else {}
            rng = random.Random(seed + 77)
            out = dict(seed=seed, p_ins=p_ins, variant=variant, tau=best, fact_accuracy=ev['FactAccuracy'], purity=ev['Purity'], bcf1=ev['BcF1'], neg_absorbed=ev['NegAbsorbed'])
            GRID = [(sm, wm) for sm in (4, 5, 6, 7, 8) for wm in (0.4, 0.5, 0.6, 0.7, 0.8)]
            f1 = lambda r, t: 2 * len(r & t) / max(1, len(r) + len(t))
            for src in ('stage1',):
                for buffered, s2 in ((False, 'plain'), (True, 'plain'), (True, 'improved')):
                    G2 = [g + (False, 0.0) for g in GRID] if s2 == 'plain' else [g + (True, c) for g in GRID for c in (0.6, 0.7, 0.8, 0.9)]
                    sm, wm, wt, cm = max(G2, key=lambda g: f1(find_rules(val_eps, va, vdec, src, buffered, *g, ENT=TRAIN_ENT), val_true))
                    rules = find_rules(eps, a, dec, src, buffered, sm, wm, wt, cm)
                    items = sorted(rules)
                    sents = generate(model, vocab, items, rng)
                    correct = {r for r, s_ in zip(items, sents) if r in true_rules and s_ == SG.target(r[0], (r[1], r[2]), False)}
                    key = f"{src}_{'buffered' if buffered else 'standard'}" + ('_improved' if s2 == 'improved' else '')
                    out[key + '_thresholds'] = [sm, wm]
                    out[key + '_rule_recall'] = len(rules & true_rules) / len(true_rules)
                    out[key + '_rule_precision'] = len(rules & true_rules) / max(1, len(rules))
                    out[key + '_sentence_accuracy'] = len(correct) / len(true_rules)
                    out[key + '_generator_exact_on_true_rules'] = len(correct) / max(1, len(rules & true_rules))
                    out[key + '_false_sentences'] = len(rules - true_rules)
            print(seed, p_ins, variant, f'{time.time()-t0:.0f}s', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if 'recall' in k or 'precision' in k or 'fact_acc' in k}, flush=True)
            results.append(out)
    json.dump(results, open(f'./e2e/e2e2_seed{seed}.json', 'w'), default=float)
    return results


if __name__ == '__main__':
    with Pool(2) as p:
        res = p.map(job, [1, 7, 42, 123, 999])
    json.dump([r for o in res for r in o], open('./e2e/e2e2_results.json', 'w'), default=float)
    print('done')
