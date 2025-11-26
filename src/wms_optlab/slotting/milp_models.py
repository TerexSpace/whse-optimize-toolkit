from typing import List, Dict
from ..data.models import SKU, Location
from ..solvers.interfaces import OptimizationBackend

def create_slotting_assignment_model(
    skus: List[SKU],
    locations: List[Location],
    costs: Dict[str, Dict[str, float]],  # cost[sku_id][loc_id]
    solver: OptimizationBackend
) -> Dict[str, str]:
    """
    Creates and solves a MILP model for the slotting assignment problem.
    This is a classic assignment problem formulation.

    Args:
        skus: List of SKUs to assign.
        locations: List of available locations.
        costs: A nested dictionary where costs[s.sku_id][l.loc_id] is the cost
               of assigning SKU s to location l (e.g., based on expected travel).
        solver: An instance of an optimization backend.

    Returns:
        A dictionary mapping SKU IDs to Location IDs, representing the optimal assignment.
    """
    # Decision variables: x[i, j] is 1 if SKU i is assigned to location j, 0 otherwise
    x = {}
    for s in skus:
        for l in locations:
            x[s.sku_id, l.loc_id] = solver.add_binary_var(f"x_{s.sku_id}_{l.loc_id}")

    # Objective function: Minimize total assignment cost
    solver.set_objective(
        solver.sum(costs[s.sku_id][l.loc_id] * x[s.sku_id, l.loc_id] for s in skus for l in locations)
    )

    # Constraint: Each SKU must be assigned to exactly one location
    for s in skus:
        solver.add_constraint(solver.sum(x[s.sku_id, l.loc_id] for l in locations) == 1)

    # Constraint: Each location can be assigned at most one SKU
    for l in locations:
        solver.add_constraint(solver.sum(x[s.sku_id, l.loc_id] for s in skus) <= 1)
        
    # TODO: Add capacity constraints if SKU volumes and location capacities are relevant

    # Solve the model
    status = solver.solve()

    # Extract the solution
    if status in (solver.OPTIMAL, solver.FEASIBLE):
        slotting_plan = {}
        for s in skus:
            for l in locations:
                if solver.get_var_value(x[s.sku_id, l.loc_id]) > 0.9:
                    slotting_plan[s.sku_id] = l.loc_id
        return slotting_plan
    else:
        raise RuntimeError("Slotting optimization failed to find a solution.")
