from typing import List, Tuple
import networkx as nx

def calculate_route_distance(route: List[str], graph: nx.Graph) -> float:
    """
    Calculates the total travel distance for a given ordered route.

    Args:
        route: An ordered list of location IDs.
        graph: The warehouse layout graph with edge weights as distances.

    Returns:
        The total distance of the route.
    """
    total_distance = 0.0
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i+1]
        try:
            # Use Dijkstra's algorithm to find the shortest path between consecutive points in the route
            total_distance += nx.shortest_path_length(graph, source=u, target=v, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Handle cases where path doesn't exist or nodes aren't in graph
            return float('inf')
    return total_distance

def find_optimal_tour_tsp(
    locations_to_visit: List[str],
    graph: nx.Graph,
    depot_loc_id: str
) -> Tuple[List[str], float]:
    """
    Finds the optimal picking tour for a set of locations using a TSP solver approach.
    This is computationally expensive and suitable for a small number of picks.
    
    NOTE: networkx has an approximate TSP solver. For exact methods, an external
    solver like OR-Tools would be needed for larger instances.

    Args:
        locations_to_visit: A list of location IDs that must be visited.
        graph: The warehouse layout graph.
        depot_loc_id: The starting and ending location for the tour.

    Returns:
        A tuple containing the optimal route (list of loc_ids) and the total distance.
    """
    nodes_in_tour = [depot_loc_id] + list(set(locations_to_visit))
    
    # Create a subgraph containing only the nodes relevant to the tour
    tour_graph = graph.subgraph(nodes_in_tour)

    # The `tsp` function in networkx provides an approximation.
    # `christofides` is a good 2-approximation algorithm for metric TSPs.
    optimal_route = nx.approximation.traveling_salesman_problem(tour_graph, cycle=True)

    # Ensure the route starts and ends at the depot
    if optimal_route[0] != depot_loc_id:
        # Find the depot's position and rotate the list
        try:
            depot_index = optimal_route.index(depot_loc_id)
            optimal_route = optimal_route[depot_index:] + optimal_route[:depot_index]
            optimal_route.append(depot_loc_id) # Close the loop
        except ValueError:
            return [], float('inf') # Depot not in route, something went wrong

    # The route from the TSP solver might not include the final leg back to the start
    if optimal_route[-1] != optimal_route[0]:
         optimal_route.append(optimal_route[0])
         
    total_distance = calculate_route_distance(optimal_route, graph)
    
    return optimal_route, total_distance
