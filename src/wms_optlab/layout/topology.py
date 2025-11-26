from typing import List, Tuple
import networkx as nx
from ..data.models import Location
from .geometry import manhattan_distance

def create_warehouse_graph(locations: List[Location], aisle_connectivity: List[Tuple[str, str]] = None) -> nx.Graph:
    """
    Creates a graph representation of the warehouse layout.

    Nodes are locations, and edges represent travel paths. Edge weights are distances.
    If aisle_connectivity is not provided, a fully connected graph is assumed.
    
    Args:
        locations: List of warehouse locations.
        aisle_connectivity: List of tuples representing connections between locations (e.g., along aisles).

    Returns:
        A networkx Graph.
    """
    G = nx.Graph()
    loc_map = {loc.loc_id: loc for loc in locations}

    for loc in locations:
        G.add_node(loc.loc_id, pos=loc.coordinates)

    if aisle_connectivity:
        for u_id, v_id in aisle_connectivity:
            if u_id in loc_map and v_id in loc_map:
                u_loc = loc_map[u_id]
                v_loc = loc_map[v_id]
                dist = manhattan_distance(u_loc.coordinates, v_loc.coordinates)
                G.add_edge(u_id, v_id, weight=dist)
    else:  # Assume fully connected for simplicity
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                u_loc = locations[i]
                v_loc = locations[j]
                dist = manhattan_distance(u_loc.coordinates, v_loc.coordinates)
                G.add_edge(u_loc.loc_id, v_loc.loc_id, weight=dist)

    return G
