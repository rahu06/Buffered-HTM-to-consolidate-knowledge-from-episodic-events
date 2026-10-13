"""Reviewer 1, Comment 1.15: Standard vs Buffer-Enhanced Temporal Memory under the Table 7 protocol and stress tests.
Training: clean sequences (no distractors), reset between sequences, 500 cycles; testing frozen (no learning).
A trial is one test sequence; the model's top-1 prediction for the next symbol is scored at every step."""
import json, random, sys
import numpy as np
sys.path.insert(0, './stage2')
from tm_buffer import TM

SEEDS = [1, 7, 42, 123, 999]
N_TRIALS = 100


def train(tm, seqs, cycles=500, rng=None, weights=None):
    for _ in range(cycles):
        order = list(range(len(seqs))); rng.shuffle(order)
        for k in order:
            seq = seqs[k] if weights is None else weights(k, rng)
            tm.reset()
            for s in seq:
                tm.step(s, learn=True)


def run_trial(tm, seq, target_idx, misleading=None):
    """Feed seq (frozen). Recovery: the event at target_idx is accepted as a correct prediction (segment or delayed
    buffered prediction) rather than bursting. Step-level: every candidate column that does not occur is a false
    prediction; a false prediction held only by the buffer is a stale-context error."""
    tm.reset(); out = dict(hit=0, interfere=0, ambiguous=0, n_cand=0, n_false=0, n_stale=0, buf_sizes=[])
    for t, s in enumerate(seq):
        seg, buf = tm.live_candidates()
        if t == target_idx:
            out['ambiguous'] = int(len(seg | buf) > 1)
            out['interfere'] = int(misleading is not None and bool((seg | buf) & misleading))
        if t > 0:
            out['n_cand'] += len(seg | buf)
            out['n_false'] += len((seg | buf) - {s})
            out['n_stale'] += len(buf - {s})
        tm.step(s, learn=False)
        if t == target_idx:
            out['hit'] = int(tm.mode in ('predicted', 'delayed'))
        out['buf_sizes'].append(tm.n_buffered())
    return out


def summarize(trials):
    n = sum(t['n_cand'] for t in trials)
    return dict(recovery=float(np.mean([t['hit'] for t in trials])),
                false_pred_rate=sum(t['n_false'] for t in trials) / max(n, 1),
                stale_rate=sum(t['n_stale'] for t in trials) / max(n, 1),
                false_per_seq=float(np.mean([t['n_false'] for t in trials])),
                stale_per_seq=float(np.mean([t['n_stale'] for t in trials])),
                ambiguity=float(np.mean([t['ambiguous'] for t in trials])),
                interference=float(np.mean([t['interfere'] for t in trials])),
                buf_mean=float(np.mean([b for t in trials for b in t['buf_sizes']])),
                buf_max=int(max(b for t in trials for b in t['buf_sizes'])))


# ---------------------------------------------------------------- vocabularies
def vocab_two_step(n_chains=15, n_dis=8):
    chains = [[2 * i, 2 * i + 1] for i in range(n_chains)]
    dis = list(range(2 * n_chains, 2 * n_chains + n_dis))
    return chains, dis, 2 * n_chains + n_dis


def experiment(buffer, seed, cond, L0=3.0):
    rng = random.Random(seed)
    res = []
    if cond.startswith('gap='):                                   # Table 7 protocol: fixed number of distractors
        k = int(cond.split('=')[1]); chains, dis, n = vocab_two_step()
        tm = TM(n, buffer=buffer, L0=L0, seed=seed); train(tm, chains, rng=rng)
        for _ in range(N_TRIALS):
            a, b = rng.choice(chains); seq = [a] + [rng.choice(dis) for _ in range(k)] + [b]
            res.append(run_trial(tm, seq, len(seq) - 1))
    elif cond == 'randomized gaps (0-6)':
        chains, dis, n = vocab_two_step(); tm = TM(n, buffer=buffer, L0=L0, seed=seed); train(tm, chains, rng=rng)
        for _ in range(N_TRIALS):
            a, b = rng.choice(chains); k = rng.randint(0, 6); seq = [a] + [rng.choice(dis) for _ in range(k)] + [b]
            res.append(run_trial(tm, seq, len(seq) - 1))
    elif cond == 'distribution shift (gaps 4-8)':
        chains, dis, n = vocab_two_step(); tm = TM(n, buffer=buffer, L0=L0, seed=seed); train(tm, chains, rng=rng)
        for _ in range(N_TRIALS):
            a, b = rng.choice(chains); k = rng.randint(4, 8); seq = [a] + [rng.choice(dis) for _ in range(k)] + [b]
            res.append(run_trial(tm, seq, len(seq) - 1))
    elif cond == 'variable-length sequences (2-5)':
        n_ch, base = 12, 0; chains = []
        for i in range(n_ch):
            L = 2 + i % 4; chains.append(list(range(base, base + L))); base += L
        dis = list(range(base, base + 8)); n = base + 8
        tm = TM(n, buffer=buffer, L0=L0, seed=seed); train(tm, chains, rng=rng)
        for _ in range(N_TRIALS):
            c = rng.choice(chains); pos = rng.randint(1, len(c) - 1); k = rng.randint(1, 3)
            seq = c[:pos] + [rng.choice(dis) for _ in range(k)] + c[pos:]
            res.append(run_trial(tm, seq, pos + k))
    elif cond.startswith('repeated symbols'):                     # A_i -> S -> B_i, two chains share each S
        chains = []; n_pairs = 6; sym = 0
        for _ in range(n_pairs):
            S = sym; sym += 1
            for _ in range(2):
                chains.append([sym, S, sym + 1]); sym += 2
        dis = list(range(sym, sym + 8)); n = sym + 8
        tm = TM(n, buffer=buffer, L0=L0, seed=seed); train(tm, chains, rng=rng)
        for _ in range(N_TRIALS):
            a, S, b = rng.choice(chains); k = rng.randint(1, 3)
            seq = [a] + [rng.choice(dis) for _ in range(k)] + [S, b]   # gap before the shared symbol
            res.append(run_trial(tm, seq, len(seq) - 1))
    elif cond == 'overlapping chains (A->B 70%, A->C 30%)':
        n_ch = 10; heads = list(range(n_ch)); B = list(range(n_ch, 2 * n_ch)); Cc = list(range(2 * n_ch, 3 * n_ch))
        dis = list(range(3 * n_ch, 3 * n_ch + 8)); n = 3 * n_ch + 8
        tm = TM(n, buffer=buffer, L0=L0, seed=seed)
        train(tm, list(range(n_ch)), rng=rng, weights=lambda k, r: [heads[k], B[k] if r.random() < 0.7 else Cc[k]])
        for _ in range(N_TRIALS):
            i = rng.randrange(n_ch); k = rng.randint(1, 3); nxt = B[i] if rng.random() < 0.7 else Cc[i]
            seq = [heads[i]] + [rng.choice(dis) for _ in range(k)] + [nxt]
            res.append(run_trial(tm, seq, len(seq) - 1))
    elif cond == 'misleading distractors (other chains\' antecedents)':
        chains, dis, n = vocab_two_step(); tm = TM(n, buffer=buffer, L0=L0, seed=seed); train(tm, chains, rng=rng)
        for _ in range(N_TRIALS):
            i = rng.randrange(len(chains)); a, b = chains[i]; k = rng.randint(1, 3)
            others = [c for j, c in enumerate(chains) if j != i]; mis = [rng.choice(others) for _ in range(k)]
            seq = [a] + [m[0] for m in mis] + [b]
            res.append(run_trial(tm, seq, len(seq) - 1, misleading={m[1] for m in mis}))
    elif cond == 'deletion (W->X->Y->Z, X missing)':
        n_ch = 8; chains = [list(range(4 * i, 4 * i + 4)) for i in range(n_ch)]; n = 4 * n_ch + 8
        tm = TM(n, buffer=buffer, L0=L0, seed=seed); train(tm, chains, rng=rng)
        for _ in range(N_TRIALS):
            w, x, y, z = rng.choice(chains); seq = [w, y, z]
            res.append(run_trial(tm, seq, 2))
    elif cond == 'substitution (X->Y->Z, Y replaced)':
        n_ch = 10; chains = [list(range(3 * i, 3 * i + 3)) for i in range(n_ch)]; dis = list(range(3 * n_ch, 3 * n_ch + 8)); n = 3 * n_ch + 8
        tm = TM(n, buffer=buffer, L0=L0, seed=seed); train(tm, chains, rng=rng)
        for _ in range(N_TRIALS):
            x, y, z = rng.choice(chains); seq = [x, rng.choice(dis), z]
            res.append(run_trial(tm, seq, 2))
    elif cond == 'context change (A->B learned, A->C at test)':
        chains, dis, n = vocab_two_step(); tm = TM(n, buffer=buffer, L0=L0, seed=seed); train(tm, chains, rng=rng)
        for _ in range(N_TRIALS):
            i = rng.randrange(len(chains)); a, b = chains[i]; c = chains[(i + 1) % len(chains)][1]; k = rng.randint(1, 3)
            seq = [a] + [rng.choice(dis) for _ in range(k)] + [c, rng.choice(dis)]
            res.append(run_trial(tm, seq, k + 1))
    return summarize(res)


CONDS = (['gap=%d' % k for k in (0, 1, 2, 3, 4, 6, 8)] +
         ['randomized gaps (0-6)', 'variable-length sequences (2-5)', 'repeated symbols (shared middle symbol, gap before it)',
          'overlapping chains (A->B 70%, A->C 30%)', "misleading distractors (other chains' antecedents)",
          'deletion (W->X->Y->Z, X missing)', 'substitution (X->Y->Z, Y replaced)',
          'context change (A->B learned, A->C at test)', 'distribution shift (gaps 4-8)'])

def work(j):
    b, s, c, L = j
    return dict(buffer=b, seed=s, cond=c, L0=L, **experiment(b, s, c, L))


if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(b, s, c, 3.0) for c in CONDS for b in (False, True) for s in SEEDS]
    jobs += [(True, s, 'randomized gaps (0-6)', L) for L in (1.0, 2.0, 5.0, 8.0) for s in SEEDS]
    jobs += [(True, s, 'context change (A->B learned, A->C at test)', L) for L in (1.0, 2.0, 5.0, 8.0) for s in SEEDS]
    jobs += [(True, s, "misleading distractors (other chains' antecedents)", L) for L in (1.0, 2.0, 5.0, 8.0) for s in SEEDS]
    with Pool(2) as p:
        out = p.map(work, jobs)
    json.dump(out, open('./stage2/stage2_results.json', 'w'), indent=1)
    print('done', len(out))
