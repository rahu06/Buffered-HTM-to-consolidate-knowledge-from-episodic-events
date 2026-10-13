"""
Cell-level HTM Temporal Memory with an optional decaying per-cell buffer (Stage 2), written from the manuscript
(Section 3.6.1, Algorithm 2). Symbols map one-to-one to columns.

Per step: (1) buffered cells decay (life *= d) and are dropped below theta_L; (2) the active column's predicted cells
are activated; if none are predicted, its buffered cells are activated as correct delayed predictions; otherwise the
column bursts and a segment is grown or reinforced (matching overlap); (3) segments that predicted a column that did
not activate are punished; (4) predicted cells whose column did not activate enter the buffer with life L0 (reset to
L0 if already buffered); (5) new predictions are made from segments whose connected overlap reaches the threshold.
This order reproduces the cutoff reported in the manuscript (recovery through 3 distractors, failure at 4).
"""
import random


class TM:
    def __init__(self, n_cols, cells=8, buffer=False, L0=3.0, d=0.6, theta_L=0.5,
                 p_conn=0.5, p_init=0.21, p_inc=0.1, p_dec=0.05, punish=0.01, act_thr=1, seed=0):
        self.n, self.C, self.buffer_on = n_cols, cells, buffer
        self.L0, self.d, self.thL = L0, d, theta_L
        self.pc, self.pi, self.inc, self.dec, self.pun, self.thr = p_conn, p_init, p_inc, p_dec, punish, act_thr
        self.rng = random.Random(seed)
        self.seg = {}            # cell -> list of segments; segment = dict presyn cell -> permanence
        self.reset()

    def reset(self):
        self.active, self.winner = set(), set()
        self.pred = {}           # cell -> connected overlap (score)
        self.pred_segs = {}      # cell -> list of active segments
        self.buf = {}            # cell -> remaining life
        self.delayed = False

    # ---------------------------------------------------------------- learning helpers
    def _reinforce(self, s, prev_active, prev_winner):
        for pre in list(s):
            s[pre] = min(1.0, s[pre] + self.inc) if pre in prev_active else max(0.0, s[pre] - self.dec)
        for w in prev_winner:
            if w not in s:
                s[w] = self.pi

    def _best_matching(self, col, prev_active):
        best, bc, bo = None, None, 0
        for i in range(self.C):
            for s in self.seg.get((col, i), []):
                o = sum(1 for pre, p in s.items() if p > 0 and pre in prev_active)
                if o > bo:
                    best, bc, bo = s, (col, i), o
        return best, bc

    # ---------------------------------------------------------------- one time step
    def step(self, col, learn=True):
        prev_active, prev_winner, prev_pred, prev_segs = self.active, self.winner, self.pred, self.pred_segs
        self.delayed = False
        if self.buffer_on:                                              # (1) decay
            self.buf = {c: l * self.d for c, l in self.buf.items() if l * self.d >= self.thL}
        cells = [c for c in prev_pred if c[0] == col]                   # (2) activate
        self.mode = 'burst'
        if cells:
            self.mode = 'predicted'
            self.active, self.winner = set(cells), set(cells)
            if learn:
                for c in cells:
                    for s in prev_segs.get(c, []):
                        self._reinforce(s, prev_active, prev_winner)
        elif self.buffer_on and any(c[0] == col for c in self.buf):
            cells = [c for c in self.buf if c[0] == col]
            for c in cells:
                del self.buf[c]
            self.active, self.winner, self.delayed = set(cells), set(cells), True
            self.mode = 'delayed'
        else:                                                           # burst
            self.active = {(col, i) for i in range(self.C)}
            if learn and prev_winner:
                s, c = self._best_matching(col, prev_active)
                if s is not None:
                    self._reinforce(s, prev_active, prev_winner)
                else:
                    counts = [len(self.seg.get((col, i), [])) for i in range(self.C)]
                    m = min(counts)
                    i = self.rng.choice([k for k, v in enumerate(counts) if v == m])
                    c = (col, i)
                    self.seg.setdefault(c, []).append({w: self.pi for w in prev_winner})
                self.winner = {c}
            else:
                self.winner = {(col, self.rng.randrange(self.C))} if not prev_winner else {(col, 0)}
        if learn:                                                       # (3) punish wrong predictions
            for c, segs in prev_segs.items():
                if c[0] != col:
                    for s in segs:
                        for pre in s:
                            if pre in prev_active:
                                s[pre] = max(0.0, s[pre] - self.pun)
        if self.buffer_on:                                              # (4) missed predictions -> buffer
            for c in prev_pred:
                if c[0] != col:
                    self.buf[c] = self.L0
        self.pred, self.pred_segs = {}, {}                              # (5) predict
        for c, segs in self.seg.items():
            for s in segs:
                o = sum(1 for pre, p in s.items() if p >= self.pc and pre in self.active)
                if o >= self.thr:
                    self.pred[c] = max(self.pred.get(c, 0), o)
                    self.pred_segs.setdefault(c, []).append(s)

    # ---------------------------------------------------------------- read-out
    def candidates(self):
        """Predicted columns (from segments) and buffered columns (from the buffer)."""
        pc, bc = {}, {}
        for c, o in self.pred.items():
            pc[c[0]] = max(pc.get(c[0], 0), o)
        for c, l in self.buf.items():
            bc[c[0]] = max(bc.get(c[0], 0), l)
        return pc, bc

    def live_candidates(self):
        """Columns the model will accept at the next step: segment predictions plus buffered cells that survive the next decay."""
        seg = {c[0] for c in self.pred}
        buf = {c[0] for c, l in self.buf.items() if l * self.d >= self.thL} if self.buffer_on else set()
        return seg, buf - seg

    def top(self):
        pc, bc = self.candidates()
        if pc:
            m = max(pc.values()); best = sorted(k for k, v in pc.items() if v == m)
            return self.rng.choice(best), 'segment'
        if bc:
            m = max(bc.values()); best = sorted(k for k, v in bc.items() if v == m)
            return self.rng.choice(best), 'buffer'
        return None, None

    def n_buffered(self):
        return len(self.buf)
