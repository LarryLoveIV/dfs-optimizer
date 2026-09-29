"""An OR-Tools backend for pydfs's public Solver interface.

Plugged in via get_optimizer(..., solver=...). Nothing in the library changes;
only the engine that searches the model.
"""
from ortools.linear_solver import pywraplp
from pydfs_lineup_optimizer.solvers.base import Solver
from pydfs_lineup_optimizer.solvers.constants import SolverSign
from pydfs_lineup_optimizer.solvers.exceptions import (SolverException,
                                                       SolverInfeasibleSolutionException)


class ORVariable:
    """A handle. pydfs uses these as dict keys, so identity must survive copy()."""
    __slots__ = ("name", "min_value", "max_value", "multiplier")

    def __init__(self, name, min_value=None, max_value=None, multiplier=None):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.multiplier = multiplier

    def __mul__(self, other):
        return ORVariable(self.name, self.min_value, self.max_value, other)

    __rmul__ = __mul__

    def resolve(self, varmap):
        var = varmap[self.name]
        return var * self.multiplier if self.multiplier is not None else var


class ORToolsSolver(Solver):
    BACKEND = "SCIP"

    def __init__(self):
        self._vars = {}
        self._constraints = []
        self._objective = None

    def setup_solver(self):
        pass

    def add_variable(self, name, min_value=None, max_value=None):
        name = name.replace(" ", "_")
        existing = self._vars.get(name)
        if existing is not None:
            return existing
        handle = ORVariable(name, min_value, max_value)
        self._vars[name] = handle
        return handle

    def set_objective(self, variables, coefficients):
        self._objective = (list(variables), list(coefficients))

    def add_constraint(self, variables, coefficients, sign, rhs, name=None):
        self._constraints.append(
            (list(variables), list(coefficients) if coefficients else None, sign, rhs, name))

    def copy(self):
        new = type(self)()
        new._vars = dict(self._vars)          # shallow: handles stay identical
        new._constraints = list(self._constraints)
        new._objective = self._objective
        return new

    def solve(self):
        model = pywraplp.Solver.CreateSolver(self.BACKEND)
        if model is None:
            raise SolverException("OR-Tools backend %s unavailable" % self.BACKEND)
        varmap = {}
        for h in self._vars.values():
            if h.min_value is not None or h.max_value is not None:
                lo = 0 if h.min_value is None else h.min_value
                hi = model.infinity() if h.max_value is None else h.max_value
                varmap[h.name] = model.IntVar(lo, hi, h.name)
            else:
                varmap[h.name] = model.BoolVar(h.name)

        for variables, coefficients, sign, rhs, _name in self._constraints:
            if coefficients:
                lhs = model.Sum([v.resolve(varmap) * c
                                 for v, c in zip(variables, coefficients)])
            else:
                lhs = model.Sum([v.resolve(varmap) for v in variables])
            right = rhs.resolve(varmap) if isinstance(rhs, ORVariable) else rhs
            if sign == SolverSign.EQ:
                model.Add(lhs == right)
            elif sign == SolverSign.GTE:
                model.Add(lhs >= right)
            elif sign == SolverSign.LTE:
                model.Add(lhs <= right)
            else:
                raise SolverException("Incorrect constraint sign")

        obj_vars, obj_coeffs = self._objective
        model.Maximize(model.Sum([v.resolve(varmap) * c
                                  for v, c in zip(obj_vars, obj_coeffs)]))
        status = model.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise SolverInfeasibleSolutionException([])
        return [h for h in self._vars.values()
                if round(varmap[h.name].solution_value() or 0) >= 1]


class ORToolsCBC(ORToolsSolver):
    BACKEND = "CBC"


class ORToolsHighs(ORToolsSolver):
    BACKEND = "HIGHS"


class ORToolsSat(ORToolsSolver):
    BACKEND = "SAT"
