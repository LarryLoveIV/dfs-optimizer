"""Correlated Monte Carlo of DraftKings NFL scores, and lineup selection on it.

Each player's score distribution comes from his projection file: Median,
Ceiling (80th percentile) and Nuke (95th) when the file has them (TwoGunGPT),
otherwise a per-position shape fitted to TwoGunGPT's boards, where those three
are near-exact linear functions of the projection. Players in the same game
are tied together by a Gaussian copula whose correlations come from real DK
scores, 2022-2025 regular seasons (nflverse, ~2,100 team-games), by role.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

# (slope, intercept) of median, p80 and p95 on the projection, by position.
SHAPE = {
    "QB": ((1.012, -1.22), (1.208, 2.34), (1.211, 10.54)),
    "RB": ((0.990, -1.34), (1.336, 1.26), (1.635, 5.86)),
    "WR": ((0.984, -1.57), (1.387, 1.09), (1.691, 6.91)),
    "TE": ((0.970, -1.34), (1.386, 0.85), (1.732, 6.26)),
    "DST": ((1.029, -0.90), (1.024, 3.45), (1.163, 7.76)),
    "K": ((0.746, 1.50), (1.247, 1.72), (1.247, 6.72)),
}
FLOOR = {"DST": -4.0}          # DST can go negative on points allowed

ROLES = ["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "TE2", "DST", "K"]
ROLE_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 2}
# Same-team correlation of DK points by role (rows/cols in ROLES order).
SAME = np.array([
    # QB1   RB1    RB2    WR1    WR2    WR3    TE1    TE2    DST    K
    [1.00,  0.06,  0.05,  0.37,  0.37,  0.21,  0.34,  0.13, -0.06,  0.05],  # QB1
    [0.06,  1.00, -0.06,  0.01,  0.01, -0.03, -0.01, -0.01,  0.06,  0.14],  # RB1
    [0.05, -0.06,  1.00,  0.05, -0.01, -0.05, -0.02,  0.01,  0.04,  0.07],  # RB2
    [0.37,  0.01,  0.05,  1.00,  0.10, -0.03, -0.01, -0.04, -0.03,  0.10],  # WR1
    [0.37,  0.01, -0.01,  0.10,  1.00,  0.02,  0.01, -0.04, -0.05,  0.02],  # WR2
    [0.21, -0.03, -0.05, -0.03,  0.02,  1.00, -0.03,  0.00, -0.09,  0.05],  # WR3
    [0.34, -0.01, -0.02, -0.01,  0.01, -0.03,  1.00, -0.03, -0.07,  0.01],  # TE1
    [0.13, -0.01,  0.01, -0.04, -0.04,  0.00, -0.03,  1.00, -0.07, -0.02],  # TE2
    [-0.06, 0.06,  0.04, -0.03, -0.05, -0.09, -0.07, -0.07,  1.00,  0.25],  # DST
    [0.05,  0.14,  0.07,  0.10,  0.02,  0.05,  0.01, -0.02,  0.25,  1.00],  # K
])
# Row role on one team vs column role on the opponent.
_OPP_RAW = np.array([
    [0.16,  0.01,  0.04,  0.11,  0.13,  0.06,  0.04,  0.07, -0.41, -0.07],
    [0.01, -0.09, -0.03, -0.02,  0.01,  0.06,  0.01, -0.01, -0.29, -0.10],
    [0.04, -0.03, -0.06, -0.00, -0.01,  0.01, -0.00,  0.05, -0.17, -0.05],
    [0.11, -0.02, -0.00,  0.10,  0.06,  0.01,  0.05,  0.01, -0.20, -0.00],
    [0.13,  0.01, -0.01,  0.06,  0.10,  0.04,  0.06,  0.08, -0.16, -0.01],
    [0.06,  0.06,  0.01,  0.01,  0.04,  0.04,  0.07,  0.01, -0.04,  0.05],
    [0.04,  0.01, -0.00,  0.05,  0.06,  0.07,  0.01,  0.00, -0.09, -0.01],
    [0.07, -0.01,  0.05,  0.01,  0.08,  0.01,  0.00,  0.06, -0.06, -0.02],
    [-0.41, -0.29, -0.17, -0.20, -0.16, -0.04, -0.09, -0.06, -0.29, -0.35],
    [-0.07, -0.10, -0.05, -0.00, -0.01,  0.05, -0.01, -0.02, -0.35, -0.14],
])
OPP = (_OPP_RAW + _OPP_RAW.T) / 2

_P = np.array([0.0, 0.5, 0.8, 0.95, 1.0])


def _norm_cdf(z):
    """Standard normal CDF (Abramowitz-Stegun 7.1.26, error < 1.5e-7)."""
    x = np.abs(z) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    erf = 1 - poly * np.exp(-x * x)
    return 0.5 * (1 + np.sign(z) * erf)


def _knots(proj, med, c80, c95, floor):
    """Quantile function knots at p = 0, .5, .8, .95, 1 for one player.

    Piecewise linear between knots. The top knot extends the 80->95 slope
    (roughly a normal tail); the bottom one is solved so the distribution's
    mean equals the projection, then clipped at the position's floor.
    """
    med, c80 = max(med, floor), max(c80, med)
    c95 = max(c95, c80)
    top = c95 + 1.2 * (c95 - c80)
    rest = 0.25 * med + 0.15 * (med + c80) + 0.075 * (c80 + c95) + 0.025 * (c95 + top)
    lo = min(max((proj - rest) / 0.25, floor), med)
    return np.array([lo, med, c80, c95, top])


def _quantile(knots, u):
    """Vectorised piecewise-linear quantile: knots (n, 5), u (n, sims)."""
    seg = np.clip(np.searchsorted(_P, u, side="right") - 1, 0, len(_P) - 2)
    p0, p1 = _P[seg], _P[seg + 1]
    rows = np.arange(knots.shape[0])[:, None]
    q0, q1 = knots[rows, seg], knots[rows, seg + 1]
    return q0 + (q1 - q0) * (u - p0) / (p1 - p0)


def assign_roles(players):
    """Role (index into ROLES) and correlation weight for each player.

    Ranked by projection within team and position. Depth beyond the tracked
    slots (WR4, RB3, backup QB...) takes the last tracked role at half weight.
    """
    role = np.zeros(len(players), dtype=int)
    weight = np.ones(len(players))
    ranked = players.assign(_i=np.arange(len(players))).sort_values("Proj", ascending=False)
    for (_, pos), grp in ranked.groupby(["Team", "Pos"], sort=False):
        for rank, i in enumerate(grp["_i"]):
            if pos in ("DST", "K"):
                role[i] = ROLES.index(pos)
                weight[i] = 1.0 if rank == 0 else 0.5
                continue
            slots = ROLE_SLOTS.get(pos)
            if slots is None:
                weight[i] = 0.0
                continue
            role[i] = ROLES.index(f"{pos}{min(rank, slots - 1) + 1}")
            weight[i] = 1.0 if rank < slots else 0.5
    return role, weight


def _game_corr(team, opp, role, weight):
    same = team[:, None] == team[None, :]
    rival = team[:, None] == opp[None, :]
    base = np.where(same, SAME[role][:, role], np.where(rival, OPP[role][:, role], 0.0))
    c = base * np.sqrt(np.outer(weight, weight))
    np.fill_diagonal(c, 1.0)
    # Nearest correlation matrix by eigenvalue clipping, so Cholesky succeeds.
    vals, vecs = np.linalg.eigh(c)
    c = (vecs * np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def simulate(players, n_sims=5000, seed=None):
    """Simulated DK points, shape (len(players), n_sims), float32.

    players: DataFrame with Name, Pos, Team, Opp, Proj and optionally
    Median, Ceiling, Nuke (used as the 50th/80th/95th percentiles).
    """
    players = players.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    have_shape = {"Median", "Ceiling", "Nuke"} <= set(players.columns)
    knots = np.empty((len(players), 5))
    for i, r in players.iterrows():
        floor = FLOOR.get(r.Pos, 0.0)
        if have_shape and pd.notna(r.Median) and pd.notna(r.Ceiling) and pd.notna(r.Nuke):
            med, c80, c95 = r.Median, r.Ceiling, r.Nuke
        else:
            (a1, b1), (a2, b2), (a3, b3) = SHAPE.get(r.Pos, SHAPE["WR"])
            med, c80, c95 = (max(a1 * r.Proj + b1, 0), a2 * r.Proj + b2, a3 * r.Proj + b3)
        knots[i] = _knots(r.Proj, med, c80, c95, floor)

    role, weight = assign_roles(players)
    z = np.empty((len(players), n_sims))
    team = players.Team.astype(str).to_numpy(dtype=object)
    opp = players.Opp.astype(str).to_numpy(dtype=object)
    game = np.array(["@".join(sorted(p)) for p in zip(team, opp)])
    for g in np.unique(game):
        idx = np.flatnonzero(game == g)
        c = _game_corr(team[idx], opp[idx], role[idx], weight[idx])
        z[idx] = np.linalg.cholesky(c) @ rng.standard_normal((len(idx), n_sims))
    u = np.clip(_norm_cdf(z), 1e-9, 1 - 1e-9)
    out = _quantile(knots, u)
    _qb_floor(out, team, role, weight)
    return out.astype(np.float32)


# A QB's passing yards and TDs are his receivers' catches, so when they score
# big he can't score small. The copula only ties them together on average, and
# left alone ~5% of its "receivers boom" sims gave the QB under 15 — never seen
# in 1,142 real starts, 2022-25. Floor = 0.45 x (WR1+WR2+WR3+TE1 - 33), which
# sits at or below the real minimum QB score at every receiver total.
QB_FLOOR_SLOPE, QB_FLOOR_START = 0.45, 33.0
CATCHER_ROLES = [ROLES.index(r) for r in ("WR1", "WR2", "WR3", "TE1")]


def _qb_floor(out, team, role, weight):
    """Raise each starting QB to the floor his receivers' scores imply, in place."""
    primary = weight == 1.0
    qb_role = ROLES.index("QB1")
    for t in np.unique(team):
        on = (team == t) & primary
        qb = np.flatnonzero(on & (role == qb_role))
        catchers = np.flatnonzero(on & np.isin(role, CATCHER_ROLES))
        if len(qb) != 1 or not len(catchers):
            continue
        floor = QB_FLOOR_SLOPE * (out[catchers].sum(axis=0) - QB_FLOOR_START)
        out[qb[0]] = np.maximum(out[qb[0]], floor)


def score_lineups(members, sims):
    """Lineup scores, shape (len(members), n_sims).

    members: per lineup, a list of (player row, multiplier) — 1.5 for a
    showdown captain, 1 otherwise.
    """
    m = np.zeros((len(members), sims.shape[0]), dtype=np.float32)
    for j, lineup in enumerate(members):
        for i, mult in lineup:
            m[j, i] += mult
    return m @ sims


def select(scores, batches, names, top_q=0.01, caps=None, thresh=None, share=None):
    """Choose lineups that together finish near the top most often.

    A sim "hits" for a lineup when it scores at least `thresh` in that sim —
    the simulated field's top-`top_q` score when there is a field, otherwise
    the top `top_q` of the candidates themselves. A hit is worth `share` of
    a win (1 / (1 + expected duplicates)): a lineup that half the field also
    played splits the prize, so equal upside favours the rarer build.

    Lineups are added one at a time, each the candidate that adds the most
    value across sims — in a sim, the set is worth its best hit — which
    maximises the chance some entry finishes near the top, what top-heavy
    payouts reward, and spreads the entries across different outcomes.

    batches: list of (candidate indices, how many to keep) — keeps each
    batch's share, so stack and flex mixes survive selection.
    names: per candidate, a set of player names, for `caps` (name -> max
    lineups). Greedy picking can corner itself — the early picks use up the
    top players' allowance and every candidate left holds one of them — so
    any overage is repaired afterwards by swapping lineups (see _repair).
    """
    caps = caps or {}
    if thresh is None:
        thresh = np.quantile(scores, 1 - top_q, axis=0)
    hits = scores >= thresh
    share = np.ones(len(scores)) if share is None else np.asarray(share, dtype=float)
    value = hits * share[:, None].astype(np.float32)
    rate = hits.mean(axis=1)
    best = np.zeros(scores.shape[1], dtype=np.float32)
    used = {}
    chosen = []
    total = sum(keep for _, keep in batches)

    def room(n, so_far):
        # Paced cap: a player may run at most one lineup ahead of his cap's
        # share of the picks so far, so the top names can't use up their
        # whole allowance in the first batch and corner the later ones.
        cap = caps.get(n)
        if cap is None:
            return True
        paced = -(-cap * (so_far + 1) // total) + 1
        return used.get(n, 0) < min(cap, paced)

    for cands, keep in batches:
        pool = list(cands)
        for _ in range(keep):
            ok = [c for c in pool if all(room(n, len(chosen)) for n in names[c])]
            ok = ok or [c for c in pool
                        if all(used.get(n, 0) < caps.get(n, 1e9) for n in names[c])]
            ok = ok or pool          # cornered: take the best, _repair fixes it
            if not ok:
                break
            gain = (np.maximum(value[ok] - best, 0).sum(axis=1)
                    + rate[ok] * share[ok])                       # breaks ties
            pick = ok[int(np.argmax(gain))]
            chosen.append(pick)
            pool.remove(pick)
            best = np.maximum(best, value[pick])
            for n in names[pick]:
                used[n] = used.get(n, 0) + 1
    over = {}
    if caps:
        chosen, over = _repair(chosen, value, names, caps, batches)
    return chosen, {"rate": rate, "hits": hits,
                    "covered": hits[chosen].any(axis=0).mean(), "over_cap": over}


def _repair(chosen, value, names, caps, batches):
    """Swap out lineups until no player is over his cap.

    Each step swaps one chosen lineup holding an over-cap player for an
    unused candidate from the same batch (so the mix holds) that puts nobody
    over, picking the swap that keeps the set worth the most. Returns the
    picks and whoever is still over, {name: lineups}, if no swap can fix it.
    """
    batch_of = {c: b for b, (cands, _) in enumerate(batches) for c in cands}
    chosen = list(chosen)
    count = Counter(n for c in chosen for n in names[c])
    for _ in range(4 * len(chosen)):
        over = {n for n, k in count.items() if k > caps.get(n, 1e9)}
        if not over:
            break
        # Per sim, the best and second-best value in the set, so dropping one
        # lineup is cheap to evaluate.
        v = value[chosen]
        top = np.argmax(v, axis=0)
        first = v[top, np.arange(v.shape[1])]
        v2 = v.copy()
        v2[top, np.arange(v.shape[1])] = -1
        second = np.maximum(v2.max(axis=0), 0) if len(chosen) > 1 else np.zeros_like(first)
        taken = set(chosen)
        pick, pick_val = None, -1.0
        for slot, out in enumerate(chosen):
            if not names[out] & over:
                continue
            base = np.where(top == slot, second, first)
            for cand in batches[batch_of[out]][0]:
                if cand in taken or any(
                        count[n] - (n in names[out]) + 1 > caps.get(n, 1e9)
                        for n in names[cand]):
                    continue
                val = np.maximum(base, value[cand]).mean()
                if val > pick_val:
                    pick, pick_val = (slot, out, cand), val
        if pick is None:
            break
        slot, out, cand = pick
        chosen[slot] = cand
        count.subtract(names[out])
        count.update(names[cand])
    return chosen, {n: k for n, k in count.items() if k > caps.get(n, 1e9)}


def coverage(scores, picks, top_q=0.01, thresh=None):
    """Share of sims in which at least one of `picks` clears `thresh` (default:
    the top `top_q` of all candidates). Run it on sims the picks weren't
    chosen on, or it flatters the selection."""
    if thresh is None:
        thresh = np.quantile(scores, 1 - top_q, axis=0)
    return (scores[picks] >= thresh).any(axis=0).mean()


# ---------------------------------------------------------------- the field
# DraftKings NFL classic. The field's habits are measured from a real 2,591-
# entry Week 3 main-slate contest: 88% of lineups pair the QB with at least one
# of his WR/TEs (40% with two), and 99% spend $49,200 or more.
CLASSIC_SLOTS = [("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1), ("DST", 1)]
FLEX_POS = ("RB", "WR", "TE")
SALARY_CAP, FIELD_MIN_SALARY = 50_000, 49_200
# Weights on the QB's own WR/TEs, his own RB and the opponent's skill players,
# tuned so the simulated field stacks, pairs and brings back as often as that
# contest did (88% / 17% / 49%).
FIELD_STACK_BOOST, FIELD_QB_RB_BOOST, FIELD_BRINGBACK_BOOST = 12.0, 2.3, 2.2


def _gumbel(rng, shape):
    return -np.log(-np.log(rng.uniform(1e-12, 1.0, shape)))


def _sample_lineups(logw, pos, team, salary, n, rng, opp_of, boost=None):
    """n field lineups, rows of 9 player indices, drawn by ownership weight.

    Each slot group is a weighted draw without replacement (Gumbel top-k),
    the QB's own pass catchers are up-weighted, and only lineups inside the
    field's salary band are kept, so fewer than n can come back.
    """
    boost = FIELD_STACK_BOOST if boost is None else boost
    qbs = np.flatnonzero(pos == "QB")
    qb = qbs[np.argmax(logw[qbs] + _gumbel(rng, (n, len(qbs))), axis=1)]
    w = np.broadcast_to(logw, (n, len(logw))).copy()
    catcher = np.isin(pos, ("WR", "TE"))[None, :]
    own = team[None, :] == team[qb][:, None]
    rival = team[None, :] == np.asarray([opp_of.get(t, "") for t in team[qb]], dtype=object)[:, None]
    w += np.log(boost) * (own & catcher)
    w += np.log(FIELD_QB_RB_BOOST) * (own & (pos == "RB")[None, :])
    w += np.log(FIELD_BRINGBACK_BOOST) * (rival & np.isin(pos, FLEX_POS)[None, :])
    picks = [qb[:, None]]
    taken = np.zeros_like(w, dtype=bool)
    taken[np.arange(n), qb] = True
    for p, k in CLASSIC_SLOTS[1:]:
        cols = np.flatnonzero(pos == p)
        keys = w[:, cols] + _gumbel(rng, (n, len(cols)))
        top = cols[np.argpartition(-keys, k - 1, axis=1)[:, :k]]
        picks.append(top)
        taken[np.arange(n)[:, None], top] = True
    cols = np.flatnonzero(np.isin(pos, FLEX_POS))
    keys = np.where(taken[:, cols], -np.inf, w[:, cols] + _gumbel(rng, (n, len(cols))))
    picks.append(cols[np.argmax(keys, axis=1)][:, None])
    lu = np.hstack(picks)
    sal = salary[lu].sum(axis=1)
    return lu[(sal <= SALARY_CAP) & (sal >= FIELD_MIN_SALARY) & np.isfinite(w[np.arange(n)[:, None], lu]).all(axis=1)]


def build_field(players, own, n=5000, seed=None, iters=8):
    """Simulated opponent lineups whose player exposures match `own` (percent).

    Weights start at the ownership numbers and are rescaled each round by
    target / observed exposure, since the salary band and stacking push some
    players above or below their raw weight. players: Pos, Team, Opp, Salary.
    Returns (lineups as an (m, 9) index array, exposure error in points).
    """
    rng = np.random.default_rng(seed)
    pos = players.Pos.astype(str).to_numpy(dtype=object)
    team = players.Team.astype(str).to_numpy(dtype=object)
    salary = players.Salary.to_numpy(dtype=float)
    opp_of = dict(zip(team, players.Opp.astype(str)))
    target = np.clip(np.nan_to_num(np.asarray(own, dtype=float)) / 100, 0, 1)
    logw = np.where(target > 0, np.log(np.maximum(target, 1e-9)), -np.inf)
    for _ in range(iters):
        lu = _sample_lineups(logw, pos, team, salary, 4 * n, rng, opp_of)
        if not len(lu):
            break
        expo = np.bincount(lu.ravel(), minlength=len(pos)) / len(lu)
        ok = (target > 0) & (expo > 0)
        logw[ok] += 0.8 * np.log(target[ok] / expo[ok])
    field = np.empty((0, 9), dtype=int)
    while len(field) < n:
        more = _sample_lineups(logw, pos, team, salary, 4 * n, rng, opp_of)
        if not len(more):
            break
        field = np.vstack([field, more])
    field = field[:n]
    expo = np.bincount(field.ravel(), minlength=len(pos)) / max(len(field), 1)
    return field, float(np.abs(expo - target)[target > 0].mean() * 100)


# Copies of a lineup that OTHER users enter, fitted (Poisson) to the real Week 3
# contest's actual ownership (DK splits %Drafted by roster slot; summed here)
# and duplicates: 0.7% of its 2,591 entries were also played by someone else,
# ~0% for the rarest fifth of builds up to ~1.7% for the chalkiest (AUC 0.80).
# Same-user repeats (one user entered a lineup 77 times) aren't splits with
# the field, so they're left out. At 150k entries this gives ~0.1 copies for
# nine 9%-owned players and ~4 for nine 25%-owned. One contest is a thin
# base; refit as more standings files come in. Showdowns use it too for now.
DUPE_SCALE, DUPE_POWER = 0.004138, 0.40
# Large fields (over 14,000 entries) use a refit on four ~85-95k-entry contests
# (Quarter Jukebox and Dime Package, Weeks 4 and 5). Fit on one week and tested
# on the other, it beat the Week 3 curve 4 for 4 (e.g. predicted 3.6% of entries
# duplicated vs 5.3% actual; the Week 3 curve said 11.1%), so it's proven there.
# Smaller fields keep the Week 3 curve until more small contests are in.
LARGE_FIELD_ENTRIES = 14_000
LARGE_DUPE_SCALE, LARGE_DUPE_POWER = 0.1353, 0.65


def expected_duplicates(lineup_own, entries):
    """Expected copies of each lineup among the other `entries` - 1 entries.

    lineup_own: per lineup, its players' ownership in percent. Rows can be
    short when a player had no projection to match; he's just left out.
    """
    q = np.array([np.prod(np.clip(np.asarray(row, dtype=float), 0.01, 100) / 100)
                  for row in lineup_own])
    scale, power = ((LARGE_DUPE_SCALE, LARGE_DUPE_POWER) if entries > LARGE_FIELD_ENTRIES
                    else (DUPE_SCALE, DUPE_POWER))
    return max(entries - 1, 0) * scale * q ** power


# ------------------------------------------------------------ showdown field
# One captain (1.5x points and salary) plus five flex from one game, both
# teams required (a DraftKings rule). Drawn from TwoGunGPT's CPT and FLEX
# ownership. The salary floor is a loose guess and the field gets no stacking
# bias beyond what the ownership implies: the one showdown contest measured
# so far (TNF Week 5) isn't enough to justify tuning either.
SHOWDOWN_FLEX, SHOWDOWN_MIN_SALARY = 5, 46_000


def _sample_showdown(cpt_w, flex_w, team, salary, n, rng):
    cpt = np.argmax(cpt_w + _gumbel(rng, (n, len(cpt_w))), axis=1)
    keys = np.broadcast_to(flex_w, (n, len(flex_w))) + _gumbel(rng, (n, len(flex_w)))
    keys[np.arange(n), cpt] = -np.inf
    flex = np.argpartition(-keys, SHOWDOWN_FLEX - 1, axis=1)[:, :SHOWDOWN_FLEX]
    ok = np.isfinite(keys[np.arange(n)[:, None], flex]).all(axis=1) & np.isfinite(cpt_w[cpt])
    sal = 1.5 * salary[cpt] + salary[flex].sum(axis=1)
    both = (team[flex] != team[cpt][:, None]).any(axis=1)
    keep = ok & both & (sal <= SALARY_CAP) & (sal >= SHOWDOWN_MIN_SALARY)
    return cpt[keep], flex[keep]


def build_showdown_field(players, cpt_own, flex_own, n=5000, seed=None, iters=8):
    """Simulated showdown opponents matching CPT and FLEX ownership (percent).

    Returns (captain index per lineup, (m, 5) flex indices, mean exposure
    error in points across both slots).
    """
    rng = np.random.default_rng(seed)
    team = players.Team.astype(str).to_numpy(dtype=object)
    salary = players.Salary.to_numpy(dtype=float)
    tc = np.clip(np.nan_to_num(np.asarray(cpt_own, dtype=float)) / 100, 0, 1)
    tf = np.clip(np.nan_to_num(np.asarray(flex_own, dtype=float)) / 100, 0, 1)
    wc = np.where(tc > 0, np.log(np.maximum(tc, 1e-9)), -np.inf)
    wf = np.where(tf > 0, np.log(np.maximum(tf, 1e-9)), -np.inf)
    for _ in range(iters):
        c, f = _sample_showdown(wc, wf, team, salary, 4 * n, rng)
        if not len(c):
            break
        ec = np.bincount(c, minlength=len(team)) / len(c)
        ef = np.bincount(f.ravel(), minlength=len(team)) / len(c)
        okc, okf = (tc > 0) & (ec > 0), (tf > 0) & (ef > 0)
        wc[okc] += 0.8 * np.log(tc[okc] / ec[okc])
        wf[okf] += 0.8 * np.log(tf[okf] / ef[okf])
    cs, fs, have = [], [], 0
    while have < n:
        c, f = _sample_showdown(wc, wf, team, salary, 4 * n, rng)
        if not len(c):
            break
        cs.append(c); fs.append(f); have += len(c)
    cpt = np.concatenate(cs)[:n] if cs else np.empty(0, dtype=int)
    flex = np.vstack(fs)[:n] if fs else np.empty((0, SHOWDOWN_FLEX), dtype=int)
    ec = np.bincount(cpt, minlength=len(team)) / max(len(cpt), 1)
    ef = np.bincount(flex.ravel(), minlength=len(team)) / max(len(cpt), 1)
    err = np.concatenate([np.abs(ec - tc)[tc > 0], np.abs(ef - tf)[tf > 0]]).mean() * 100
    return cpt, flex, float(err)


# ------------------------------------------------------------- optimal rate
def optimal_rates(players, sims, n=500, showdown=False):
    """How often each player is in the best legal lineup of a simulated game.

    For each of the first `n` sims, solves for that sim's highest-scoring
    DraftKings lineup (salary cap and roster rules, nothing else) and counts
    who is in it. Set beside projected ownership, it's the usual leverage
    read: a player in the optimal lineup far more often than the field plays
    him is underowned. Showdown also returns the captain share.

    players: Pos, Team, Salary (FLEX salary on a showdown). Returns
    (optimal % per player, captain % per player or None).
    """
    from ortools.sat.python import cp_model

    sal = np.nan_to_num(players.Salary.to_numpy(dtype=float), nan=1e9).astype(int)
    pos = players.Pos.astype(str).to_numpy()
    team = players.Team.astype(str).to_numpy()
    ok = sal < SALARY_CAP
    idx = np.flatnonzero(ok)
    m = cp_model.CpModel()
    x = {i: m.NewBoolVar(f"x{i}") for i in idx}
    if showdown:
        c = {i: m.NewBoolVar(f"c{i}") for i in idx}
        m.Add(sum(c.values()) == 1)
        m.Add(sum(x.values()) == SHOWDOWN_FLEX)
        for i in idx:
            m.Add(x[i] + c[i] <= 1)
        m.Add(sum(3 * sal[i] * c[i] + 2 * sal[i] * x[i] for i in idx) <= 2 * SALARY_CAP)
        for t in np.unique(team[idx]):
            m.Add(sum(x[i] + c[i] for i in idx if team[i] == t) >= 1)
    else:
        c = None
        count = lambda p: sum(x[i] for i in idx if pos[i] == p)
        m.Add(count("QB") == 1)
        m.Add(count("DST") == 1)
        m.Add(count("RB") >= 2)
        m.Add(count("WR") >= 3)
        m.Add(count("TE") >= 1)
        m.Add(count("RB") + count("WR") + count("TE") == 7)
        m.Add(sum(sal[i] * x[i] for i in idx) <= SALARY_CAP)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 8
    solver.parameters.max_time_in_seconds = 5
    hits = np.zeros(len(players))
    cap_hits = np.zeros(len(players))
    n = min(n, sims.shape[1])
    for s in range(n):
        pts = np.round(sims[:, s] * 100).astype(int)
        obj = sum(int(pts[i]) * x[i] for i in idx)
        if showdown:
            obj += sum(int(round(1.5 * pts[i])) * c[i] for i in idx)
        m.Maximize(obj)
        if solver.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        for i in idx:
            if solver.Value(x[i]):
                hits[i] += 1
            if showdown and solver.Value(c[i]):
                hits[i] += 1
                cap_hits[i] += 1
    return hits / n * 100, (cap_hits / n * 100 if showdown else None)
