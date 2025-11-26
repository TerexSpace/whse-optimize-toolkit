from typing import Dict, List
import networkx as nx
from collections import Counter
from ..data.models import Order, Location

def calculate_expected_travel_distance(
    slotting_plan: Dict[str, str],  # {sku_id: loc_id}
    orders: List[Order],
    warehouse_graph: nx.Graph,
    depot_loc_id: str
) -> float:
    """
    Calculates the total expected travel distance for a given slotting plan and order profile.
    This assumes a simple routing policy where the picker travels from the depot
    to each required location and back to the depot for each order (single-item orders).
    More complex routing is handled in the `routing` module.

    Args:
        slotting_plan: A dictionary mapping SKU IDs to Location IDs.
        orders: A list of orders.
        warehouse_graph: A networkx graph of the warehouse layout with edge weights as distances.
        depot_loc_id: The ID of the depot location.

    Returns:
        The total travel distance.
    """
    total_distance = 0.0
    
    # Pre-calculate shortest path distances from the depot to all other locations
    try:
        dist_from_depot = nx.shortest_path_length(warehouse_graph, source=depot_loc_id, weight='weight')
    except nx.NetworkXNoPath:
        # Handle disconnected components if necessary
        return float('inf')

    # Calculate demand for each SKU
    sku_demand = Counter()
    for order in orders:
        for line in order.order_lines:
            sku_demand[line.sku.sku_id] += line.quantity

    # Calculate total distance based on demand and slotting
    for sku_id, demand in sku_demand.items():
        if sku_id in slotting_plan:
            loc_id = slotting_plan[sku_id]
            if loc_id in dist_from_depot:
                # Assuming simple back-and-forth travel for each pick
                # A round trip from the depot to the location
                round_trip_dist = 2 * dist_from_depot[loc_id]
                total_distance += demand * round_trip_dist

    return total_distance
