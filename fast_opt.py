"""Faster lineup building with the SAME lineups (look-ahead engine; deeper look-ahead on idle cores since 2026-11-14).

What it does
------------
The library builds lineups strictly one after another: each solve waits for the
previous one because the previous lineup adds rules (stay 2 players different,
exposure caps filling up). Each solve is a hard search that takes ~0.15 s.

Here the upcoming lineups are solved ahead of time on the other CPU cores, using
the rules known so far and the exact random projections the library is about to
draw for them. When the library reaches that lineup, the early answer is kept
only if it is provably the answer the normal solve would give:

  * same projections (objective) as the real solve,
  * every rule the early solve used is still in force (nothing was loosened),
  * the early answer also obeys every rule added since.

A best answer under fewer rules that also obeys the extra rules is the best
answer under all of them, so the lineup is identical. Anything else is thrown
away and solved the normal way. With a fixed random seed the output is the same
lineups in the same order as the current app.

Use: `from fast_opt import FastSat as FAST_SOLVER` (importing patches
LineupOptimizer.optimize; other solver classes keep the library's own loop).
"""
import os
import random
import time
from collections import Counter, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from ortools.linear_solver import pywraplp
from pydfs_lineup_optimizer import LineupOptimizer
from pydfs_lineup_optimizer.context import OptimizationContext
from pydfs_lineup_optimizer.exceptions import GenerateLineupException
from pydfs_lineup_optimizer.fantasy_points_strategy import (RandomFantasyPointsStrategy,
                                                            StandardFantasyPointsStrategy)
from pydfs_lineup_optimizer.rules import Objective
from pydfs_lineup_optimizer.solvers.constants import SolverSign
from pydfs_lineup_optimizer.solvers.exceptions import (SolverException,
                                                       SolverInfeasibleSolutionException)

from or_solver import ORToolsSolver, ORVariable

AHEAD = int(os.environ.get("FAST_AHEAD", "10"))            # lineups always solved ahead; 0 = off (current engine)
WORKERS = int(os.environ.get("FAST_WORKERS", "10"))        # early solves running at once
DEEP = int(os.environ.get("FAST_DEEP", "16"))             # how far ahead to go when cores would otherwise sit idle
SPILL = int(os.environ.get("FAST_SPILL", "2"))            # extra solves allowed at once for the nearest lineups
SPEC_PARAMS = os.environ.get("FAST_SPEC_PARAMS", "num_workers:1,cp_model_presolve:false")
MAIN_PARAMS = os.environ.get("FAST_MAIN_PARAMS", "num_workers:1,cp_model_presolve:false")
STATS = Counter()
TIMES = Counter()      # seconds by phase
RECORD = bool(os.environ.get("FAST_RECORD"))   # test harness only
PICKS = []          # every solve's answer, in order (for the identical-lineups test)
_POOL = None
STATS_LAST = [None]


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=WORKERS + SPILL + 1)
    return _POOL


def _solve_values(vars_, constraints, objective, params):
    """Build the model and solve it. Returns {variable name: value}, or None if infeasible."""
    model = pywraplp.Solver.CreateSolver("SAT")
    if model is None:
        raise SolverException("OR-Tools SAT backend unavailable")
    t0 = time.perf_counter()
    inf = model.infinity()
    varmap = {}
    for h in vars_.values():
        if h.min_value is not None or h.max_value is not None:
            lo = 0 if h.min_value is None else h.min_value
            hi = inf if h.max_value is None else h.max_value
            varmap[h.name] = model.IntVar(lo, hi, h.name)
        else:
            varmap[h.name] = model.BoolVar(h.name)
    for variables, coefficients, sign, rhs, _name in constraints:
        # Gather each variable's total coefficient (a variable can appear twice, and the
        # right-hand side can itself be a variable, which moves to the left).
        row = {}
        const = 0.0
        if coefficients:
            pairs = zip(variables, coefficients)
        else:
            pairs = ((v, 1) for v in variables)
        for v, c in pairs:
            if isinstance(v, ORVariable):
                row[v.name] = row.get(v.name, 0) + (c if v.multiplier is None else c * v.multiplier)
            else:
                const += v * c
        if isinstance(rhs, ORVariable):
            row[rhs.name] = row.get(rhs.name, 0) - (1 if rhs.multiplier is None else rhs.multiplier)
            right = -const
        else:
            right = rhs - const
        if sign == SolverSign.EQ:
            ct = model.Constraint(right, right)
        elif sign == SolverSign.GTE:
            ct = model.Constraint(right, inf)
        elif sign == SolverSign.LTE:
            ct = model.Constraint(-inf, right)
        else:
            raise SolverException("Incorrect constraint sign")
        for name, c in row.items():
            ct.SetCoefficient(varmap[name], c)
    obj_vars, obj_coeffs = objective
    objective_ = model.Objective()
    total = {}
    for v, c in zip(obj_vars, obj_coeffs):
        total[v.name] = total.get(v.name, 0) + (c if v.multiplier is None else c * v.multiplier)
    for name, c in total.items():
        objective_.SetCoefficient(varmap[name], c)
    objective_.SetMaximization()
    TIMES["model setup (Python)"] += time.perf_counter() - t0
    if params:
        model.SetSolverSpecificParametersAsString(params)
    t0 = time.perf_counter()
    status = model.Solve()
    TIMES["search (all cores, summed)"] += time.perf_counter() - t0
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None
    return {name: var.solution_value() or 0 for name, var in varmap.items()}


class FastSat(ORToolsSolver):
    """Same model as or_solver.ORToolsSat; marks the optimizer for the look-ahead loop."""
    BACKEND = "SAT"

    # solve() is only reached when the look-ahead loop is not used (no randomness, e.g.
    # cash mode). It is inherited unchanged from the current engine, so exact ties between
    # equal-projection lineups break the same way they do today.


def _term(v):
    return (v.name, v.multiplier) if isinstance(v, ORVariable) else ("#", v)


def _atoms(cons):
    """Rules as comparable (key, rule) pairs. A rule saying "none of these players" is split
    into one rule per player, so a block list that grew still contains the old one."""
    out = []
    for con in cons:
        variables, coefficients, sign, rhs, name = con
        if (sign == SolverSign.EQ and not coefficients and rhs == 0 and not isinstance(rhs, ORVariable)
                and all(isinstance(v, ORVariable) and v.multiplier is None for v in variables)):
            for v in variables:
                out.append((("block", v.name), ([v], None, sign, 0, name)))
        else:
            out.append(((tuple(_term(v) for v in variables),
                         tuple(coefficients) if coefficients else None, sign, _term(rhs)), con))
    return out


def _holds(con, values):
    variables, coefficients, sign, rhs, _name = con

    def val(v):
        if isinstance(v, ORVariable):
            x = values[v.name]
            return x * v.multiplier if v.multiplier is not None else x
        return v
    if coefficients:
        lhs = sum(val(v) * c for v, c in zip(variables, coefficients))
    else:
        lhs = sum(val(v) for v in variables)
    right = val(rhs)
    if sign == SolverSign.EQ:
        return abs(lhs - right) < 1e-6
    if sign == SolverSign.GTE:
        return lhs >= right - 1e-6
    return lhs <= right + 1e-6


_library_optimize = LineupOptimizer.optimize


def optimize(self, n, max_exposure=None, randomness=False, with_injured=None,
             exposure_strategy=None, exclude_lineups=None):
    """LineupOptimizer.optimize with look-ahead solving. Same lineups, same order."""
    kwargs = dict(max_exposure=max_exposure, randomness=randomness, with_injured=with_injured,
                  exclude_lineups=exclude_lineups)
    if exposure_strategy is not None:
        kwargs["exposure_strategy"] = exposure_strategy
    strategy = self.fantasy_points_strategy
    if (self._solver_class is not FastSat or randomness or with_injured is not None or AHEAD < 1
            or type(strategy) is not RandomFantasyPointsStrategy):
        yield from _library_optimize(self, n, **kwargs)
        return

    # --- setup: identical to the library's ---
    players = self.player_pool.filtered_players
    ctx_kwargs = dict(total_lineups=n, players=players, max_exposure=max_exposure,
                      randomness=randomness, exclude_lineups=exclude_lineups or [])
    if exposure_strategy is not None:
        ctx_kwargs["exposure_strategy"] = exposure_strategy
    context = OptimizationContext(**ctx_kwargs)
    rules = self._rules.copy()
    rules.update(self.settings.extra_rules)
    rules.add(Objective)
    base_solver = self._solver_class()
    base_solver.setup_solver()
    players_dict = OrderedDict(
        [(player, base_solver.add_variable(base_solver.build_player_var_name(player, str(i))))
         for i, player in enumerate(players)])
    variables_dict = {v: k for k, v in players_dict.items()}
    constraints = [constraint(self, players_dict, context) for constraint in rules]
    for constraint in constraints:
        constraint.apply(base_solver)
    n_base = len(base_solver._constraints)
    get_points = strategy.get_player_fantasy_points
    player_list = list(players_dict)
    pool = _pool()
    pending = {}                    # lineup number -> (future, objective coefficients, rule keys, var names)
    cache = {}                      # predicted projections for upcoming lineups (see _predict)
    previous_lineup = None
    try:
        for it in range(n):
            t0 = time.perf_counter()
            solver = base_solver.copy()
            for constraint in constraints:
                constraint.apply_for_iteration(solver, previous_lineup)
            TIMES["library rules per lineup"] += time.perf_counter() - t0
            # Rules the library added for this lineup, beyond the fixed ones.
            if solver._constraints[:n_base] != base_solver._constraints:
                extra_now = None     # base rules changed under us: don't trust look-ahead
            else:
                extra_now = solver._constraints[n_base:]
            true_obj = tuple(solver._objective[1])
            values = None
            job = pending.pop(it, None)
            atoms = _atoms(extra_now) if extra_now is not None else None
            now_keys = Counter(k for k, _ in atoms) if atoms is not None else None
            look = None
            if extra_now is not None:
                look = dict(pool=pool, pending=pending, it=it, n=n, solver=solver, atoms=atoms,
                            keys=now_keys, get_points=get_points, player_list=player_list,
                            cache=cache, snap=None, head=None)
            head = None
            if job is not None and extra_now is not None:
                future, obj, keys, names = job
                if obj != true_obj or names != set(solver._vars):
                    STATS["projections or variables changed"] += 1
                    cache.clear()
                elif keys - now_keys:
                    STATS["a rule was loosened"] += 1
                    for k in (keys - now_keys):
                        STATS["  loosened: " + ("block" if k[0] == "block" else str(k[2]))] += 1
                else:
                    head = future
            if look is not None:
                look["head"] = head
                _look_ahead(look, True)
            if head is not None:
                t0 = time.perf_counter()
                early = _wait(head, look)
                TIMES["waiting for an early solve"] += time.perf_counter() - t0
                if early is None:
                    STATS["early solve had no answer"] += 1
                else:
                    added = now_keys - keys
                    bad = [c for k, c in atoms if k in added and not _holds(c, early)]
                    if not bad:
                        values = early
                        STATS["kept"] += 1
                        STATS_LAST[0] = "kept"
                    else:
                        STATS["broke a newer rule"] += 1
                        for c in bad:
                            STATS["  broke: " + str(c[4]).rstrip("0123456789")] += 1
            if values is None:
                STATS["solved normally"] += 1
                STATS_LAST[0] = "normal"
                t0 = time.perf_counter()
                # Same solve as before, run on a pool thread so further look-ahead can start meanwhile.
                normal = pool.submit(_solve_values, solver._vars, solver._constraints, solver._objective,
                                     MAIN_PARAMS)
                if look is not None:
                    look["head"] = normal
                values = _wait(normal, look)
                TIMES["normal solves (wall)"] += time.perf_counter() - t0
                if values is None:
                    raise GenerateLineupException(
                        SolverInfeasibleSolutionException([]).get_user_defined_constraints())
            solved_variables = [h for h in solver._vars.values() if round(values[h.name]) >= 1]
            if RECORD:
                PICKS.append(sorted(h.name for h in solved_variables))
                if os.environ.get("FAST_DUMP") and len(PICKS) - 1 == int(os.environ["FAST_DUMP"]):
                    import pickle
                    pickle.dump(dict(vars=solver._vars, cons=solver._constraints, obj=solver._objective, values=values,
                                     kept=head is not None and values is not None and STATS_LAST[0] == "kept",
                                     it=it, n=n), open("out/dump_%s.pkl" % os.environ["FAST_DUMP"], "wb"))
            lineup_players = []
            variables_names = []
            for solved_variable in solved_variables:
                player = variables_dict.get(solved_variable)
                if player:
                    lineup_players.append(player)
                variables_names.append(solved_variable.name)
            lineup = self._build_lineup(lineup_players, context)
            previous_lineup = lineup
            context.add_lineup(lineup)
            yield lineup
            total_players = self.player_pool.total_players
            if total_players and len(self.player_pool.locked_players) == total_players:
                return
            for constraint in constraints:
                constraint.post_optimize(variables_names)
        self.last_context = context
    finally:
        for future, *_ in pending.values():
            future.cancel()


def _wait(head, look):
    """Wait for the solve the build needs next. While waiting, each time any solve finishes,
    start look-ahead solves further out on the freed core."""
    if look is None:
        return head.result()
    while not head.done():
        running = [f for f, *_ in look["pending"].values() if not f.done()]
        running.append(head)
        wait(running, return_when=FIRST_COMPLETED)
        if not head.done():
            _look_ahead(look, False)
    return head.result()


def _predict(look, last):
    """The projections the library will draw for lineups it+1..last. They are drawn now and the
    random generator is rewound, so the library draws the very same ones. Earlier predictions are
    reused only if the generator is exactly where they said it would be."""
    cache, it = look["cache"], look["it"]
    state = random.getstate()
    if cache.get("at") != it or cache.get("state") != state:
        cache.clear()
        cache.update(at=it, state=state, obj={}, end=state, upto=it)
    if cache["upto"] < last:
        random.setstate(cache["end"])
        get_points, player_list = look["get_points"], look["player_list"]
        for j in range(cache["upto"] + 1, last + 1):
            cache["obj"][j] = tuple(get_points(p) for p in player_list)
        cache["end"] = random.getstate()
        cache["upto"] = last
        random.setstate(state)
    return cache["obj"]


def _look_ahead(look, new_rules):
    t0 = time.perf_counter()
    try:
        return _look_ahead_(look, new_rules)
    finally:
        TIMES["look-ahead bookkeeping"] += time.perf_counter() - t0


def _look_ahead_(look, new_rules):
    """Start early solves for upcoming lineups that don't have one, nearest first, and (when the
    rules just changed) restart any finished early solve the rules known now have ruled out.
    The next AHEAD lineups always get one; beyond that, up to DEEP, only on a core that is free."""
    pool, pending, it, n, solver = look["pool"], look["pending"], look["it"], look["n"], look["solver"]
    atoms, keys = look["atoms"], look["keys"]
    near = min(it + AHEAD, n - 1)
    far = min(it + max(AHEAD, DEEP), n - 1)
    if far <= it:
        return
    names = set(solver._vars)
    busy = sum(1 for f, *_ in pending.values() if not f.done())
    if look["head"] is not None and not look["head"].done():
        busy += 1
    todo = []
    for j in range(it + 1, far + 1):
        job = pending.get(j)
        if job is None:
            todo.append(j)
            continue
        if not new_rules:
            continue
        future, _obj, old_keys, old_names = job
        if not future.done():
            continue
        early = future.result()
        if old_names != names or (old_keys - keys):
            stale = True
        elif early is None:
            stale = False                      # no answer under fewer rules: none under more
        else:
            added = keys - old_keys
            stale = any(k in added and not _holds(c, early) for k, c in atoms)
        if stale:
            todo.append(j)
            STATS["restarted early solve"] += 1
    start = []
    for j in todo:                              # nearest first
        if busy >= (WORKERS + SPILL if j <= near else WORKERS):
            if j > near:
                break
            continue
        start.append(j)
        busy += 1
    # A stale answer that could not be restarted now must not be mistaken for a fresh one later.
    for j in todo:
        if j not in start and j in pending:
            del pending[j]
    if not start:
        return
    objectives = _predict(look, far)
    if look["snap"] is None:
        look["snap"] = (dict(solver._vars), list(solver._constraints), list(solver._objective[0]))
    vars_, cons, obj_vars = look["snap"]
    for j in start:
        future = pool.submit(_solve_values, vars_, cons, (obj_vars, objectives[j]), SPEC_PARAMS)
        pending[j] = (future, objectives[j], keys, names)
        STATS["started early"] += 1
        if j > near:
            STATS["  started beyond the usual window"] += 1


LineupOptimizer.optimize = optimize
