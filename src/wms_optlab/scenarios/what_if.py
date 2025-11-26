from typing import List, Dict, Any
from copy import deepcopy
from ..data.models import Warehouse, Order
from ..slotting.evaluation import calculate_expected_travel_distance
from ..routing.simulation_interface import simulate_picking_process

def run_what_if_analysis(
    base_warehouse: Warehouse,
    base_orders: List[Order],
    scenarios: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Runs a "what-if" analysis for different warehouse configurations or order profiles.

    Args:
        base_warehouse: The baseline warehouse model.
        base_orders: The baseline list of orders.
        scenarios: A dictionary where each key is a scenario name and the value is a
                   dictionary defining the changes for that scenario (e.g., new slotting).

    Example Scenarios:
    scenarios = {
        "Scenario A: Popularity Slotting": {
            "slotting_plan": slotting_plan_A,
            "routing_policy": "s_shape"
        },
        "Scenario B: MILP Slotting": {
            "slotting_plan": slotting_plan_B,
            "routing_policy": "s_shape"
        }
    }

    Returns:
        A dictionary with the evaluation results for each scenario.
    """
    results = {}
    
    # Assume a depot and graph are part of the warehouse object or can be derived
    warehouse_graph = base_warehouse.get_graph() 
    depot = next(l for l in base_warehouse.locations if l.location_type == 'depot')

    for name, params in scenarios.items():
        warehouse_scenario = deepcopy(base_warehouse)
        orders_scenario = deepcopy(base_orders)

        # Modify warehouse or orders based on scenario parameters
        # (e.g., change item locations based on a new slotting plan)
        
        slotting_plan = params.get("slotting_plan")
        if not slotting_plan:
            results[name] = {"error": "No slotting plan provided for scenario."}
            continue

        # Use the simulation interface for a more complete evaluation
        sim_results = simulate_picking_process(
            warehouse=warehouse_scenario,
            orders=orders_scenario,
            slotting_plan=slotting_plan,
            routing_policy=params.get("routing_policy", "s_shape")
        )
        
        # Or use a simpler metric like expected travel distance
        expected_dist = calculate_expected_travel_distance(
            slotting_plan=slotting_plan,
            orders=orders_scenario,
            warehouse_graph=warehouse_graph,
            depot_loc_id=depot.loc_id
        )

        results[name] = {
            "expected_travel_distance": expected_dist,
            "simulation_results": sim_results
        }
        
    return results
