from typing import List, Tuple
import networkx as nx
from ..data.models import Location

def get_s_shape_route(
    pick_locations: List[Location],
    warehouse_graph: nx.Graph,
    depot_loc: Location
) -> List[str]:
    """
    Generates a picker route using the S-shape (or serpentine) heuristic.
    Assumes a standard grid layout where aisles are primarily oriented along one axis.

    Args:
        pick_locations: A list of locations to visit for an order.
        warehouse_graph: The warehouse layout graph.
        depot_loc: The starting and ending depot location.

    Returns:
        An ordered list of location IDs representing the route.
    """
    if not pick_locations:
        return [depot_loc.loc_id, depot_loc.loc_id]

    route = [depot_loc.loc_id]
    
    # Group locations by aisle (assuming aisle is encoded in x-coordinate)
    aisles = {}
    for loc in pick_locations:
        aisle_x = loc.coordinates[0]
        if aisle_x not in aisles:
            aisles[aisle_x] = []
        aisles[aisle_x].append(loc)

    # Sort aisles by their x-coordinate
    sorted_aisle_keys = sorted(aisles.keys())

    # Traverse aisles in an S-shape pattern
    for i, aisle_x in enumerate(sorted_aisle_keys):
        aisle_locs = aisles[aisle_x]
        
        # Sort locations within the aisle by y-coordinate
        # Direction depends on whether the aisle index is even or odd
        direction_ascending = (i % 2 == 0)
        aisle_locs.sort(key=lambda loc: loc.coordinates[1], reverse=not direction_ascending)
        
        route.extend([loc.loc_id for loc in aisle_locs])

    route.append(depot_loc.loc_id)
    return route
