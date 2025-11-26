import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List
from ..data.models import Location

def plot_warehouse_layout(
    graph: nx.Graph,
    slotting_plan: Dict[str, str] = None,
    route: List[str] = None,
    title: str = "Warehouse Layout"
):
    """
    Generates a 2D plot of the warehouse layout.

    Args:
        graph: A networkx graph representing the warehouse topology.
        slotting_plan: Optional dictionary mapping SKU IDs to Location IDs to color-code locations.
        route: Optional list of location IDs representing a picker's route to highlight.
        title: The title of the plot.
    """
    pos = nx.get_node_attributes(graph, 'pos')
    
    # Use 2D projection (x, y)
    pos_2d = {node: (coords[0], coords[1]) for node, coords in pos.items()}

    plt.figure(figsize=(12, 8))
    
    node_colors = 'skyblue'
    if slotting_plan:
        # Simple coloring: just distinguish assigned vs. unassigned
        assigned_locs = set(slotting_plan.values())
        node_colors = ['green' if node in assigned_locs else 'lightgray' for node in graph.nodes()]

    nx.draw(
        graph,
        pos=pos_2d,
        with_labels=True,
        node_size=500,
        node_color=node_colors,
        font_size=8,
        font_color='black',
        edge_color='gray'
    )

    if route:
        route_edges = list(zip(route, route[1:]))
        nx.draw_networkx_edges(
            graph,
            pos=pos_2d,
            edgelist=route_edges,
            width=2.5,
            edge_color='red',
            style='dashed'
        )

    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True)
    plt.show()
