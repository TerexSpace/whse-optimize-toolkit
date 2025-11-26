from abc import ABC, abstractmethod
from typing import Any, Dict

class OptimizationBackend(ABC):
    """
    Abstract base class for a generic optimization solver backend.
    This interface abstracts away the specifics of libraries like OR-Tools or PuLP.
    """
    # Solver status constants
    OPTIMAL = 'OPTIMAL'
    FEASIBLE = 'FEASIBLE'
    INFEASIBLE = 'INFEASIBLE'
    UNBOUNDED = 'UNBOUNDED'
    ERROR = 'ERROR'

    @abstractmethod
    def add_binary_var(self, name: str) -> Any:
        """Adds a binary (0-1) decision variable to the model."""
        pass

    @abstractmethod
    def add_continuous_var(self, name: str, lb: float = 0, ub: float = float('inf')) -> Any:
        """Adds a continuous decision variable."""
        pass
    
    @abstractmethod
    def set_objective(self, expression: Any, sense: str = 'minimize'):
        """Sets the objective function for the model."""
        pass

    @abstractmethod
    def add_constraint(self, expression: Any):
        """Adds a constraint to the model."""
        pass

    @abstractmethod
    def solve(self) -> str:
        """Solves the optimization model."""
        pass

    @abstractmethod
    def get_var_value(self, var: Any) -> float:
        """Gets the value of a decision variable after solving."""
        pass

    @abstractmethod
    def sum(self, terms: list) -> Any:
        """Creates a sum expression for use in objectives or constraints."""
        pass
