from typing import List, Dict, Any
from ..data.models import Order
from ..routing.policies import get_s_shape_route
from ..routing.shortest_path import calculate_route_distance
from ..data.models import Warehouse

def simulate_picking_process(
    warehouse: Warehouse,
    orders: List[Order],
    slotting_plan: Dict[str, str], # {sku_id: loc_id}
    routing_policy: str = 's_shape'
) -> Dict[str, Any]:
    """
    A simple simulation stub to evaluate picking routes for a set of orders.
    This is not a discrete-event simulation, but a static analysis of routes.

    Args:
        warehouse: The warehouse object with layout and item data.
        orders: The list of orders to be "picked".
        slotting_plan: The current assignment of SKUs to locations.
        routing_policy: The name of the routing policy to use (e.g., 's_shape').

    Returns:
        A dictionary with simulation results, like total distance and number of picks.
    """
    total_distance = 0.0
    total_picks = 0
    
    warehouse_graph = warehouse.get_graph() # Assuming a method to get the graph
    depot_loc = next((loc for loc in warehouse.locations if loc.location_type == 'depot'), None)
    
    if not depot_loc:
        raise ValueError("Depot location not found in warehouse.")

    loc_map = {loc.loc_id: loc for loc in warehouse.locations}

    for order in orders:
        pick_locations_ids = [slotting_plan.get(line.sku.sku_id) for line in order.order_lines]
        pick_locations = [loc_map[loc_id] for loc_id in pick_locations_ids if loc_id and loc_id in loc_map]
        
        if not pick_locations:
            continue
            
        total_picks += len(pick_locations)
        
        route = []
        if routing_policy == 's_shape':
            route = get_s_shape_route(pick_locations, warehouse_graph, depot_loc)
        else:
            # Default to a simple naive route for other policies for now
            route = [depot_loc.loc_id] + [loc.loc_id for loc in pick_locations] + [depot_loc.loc_id]
            
        order_distance = calculate_route_distance(route, warehouse_graph)
        total_distance += order_distance

    return {
        "total_distance": total_distance,
        "number_of_orders": len(orders),
        "number_of_picks": total_picks,
        "average_distance_per_order": total_distance / len(orders) if orders else 0
    }
