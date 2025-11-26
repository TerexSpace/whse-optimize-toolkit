from ortools.sat.python import cp_model
from .interfaces import OptimizationBackend

class ORToolsBackend(OptimizationBackend):
    """
    An implementation of the OptimizationBackend interface using Google's OR-Tools CP-SAT solver.
    """
    def __init__(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self._vars = {}
        self.status_map = {
            cp_model.OPTIMAL: self.OPTIMAL,
            cp_model.FEASIBLE: self.FEASIBLE,
            cp_model.INFEASIBLE: self.INFEASIBLE,
            cp_model.UNKNOWN: self.ERROR,
            cp_model.MODEL_INVALID: self.ERROR,
        }

    def add_binary_var(self, name: str):
        var = self.model.NewBoolVar(name)
        self._vars[name] = var
        return var

    def add_continuous_var(self, name: str, lb: float = 0, ub: float = float('inf')):
        # The CP-SAT solver works with integers, so we need to be careful here.
        # For true continuous variables, a different solver like GLOP or a MIP solver would be better.
        # This is a simplification for assignment-type problems.
        if isinstance(lb, float) or isinstance(ub, float):
            # For non-integer bounds, we would need to scale and discretize.
            # This is a limitation of this specific backend choice for general LP problems.
            # We will assume integer bounds for now.
            pass
        var = self.model.NewIntVar(int(lb), int(ub), name)
        self._vars[name] = var
        return var
    
    def set_objective(self, expression, sense: str = 'minimize'):
        if sense.lower() == 'minimize':
            self.model.Minimize(expression)
        elif sense.lower() == 'maximize':
            self.model.Maximize(expression)
        else:
            raise ValueError("Sense must be 'minimize' or 'maximize'")

    def add_constraint(self, expression):
        self.model.Add(expression)

    def solve(self) -> str:
        self.status = self.solver.Solve(self.model)
        return self.status_map.get(self.status, self.ERROR)

    def get_var_value(self, var):
        if self.status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self.solver.Value(var)
        return 0.0

    def sum(self, terms: list):
        return cp_model.LinearExpr.Sum(terms)
