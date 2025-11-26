from typing import List, Dict
import pandas as pd
from collections import Counter

def calculate_picker_workload(
    order_assignments: Dict[str, List[str]] # {picker_id: [order_id_1, ...]}
) -> Dict[str, int]:
    """Calculates workload as the number of orders assigned to each picker."""
    return {picker: len(orders) for picker, orders in order_assignments.items()}

def calculate_congestion_proxy(
    routes: List[List[str]] # List of routes, where each route is a list of loc_ids
) -> Counter:
    """
    Calculates a simple proxy for congestion by counting how many times
    each location (or aisle) is visited across all routes.
    """
    location_visits = Counter()
    for route in routes:
        location_visits.update(route)
    return location_visits

def get_kpis_as_dataframe(
    simulation_results: Dict
) -> pd.DataFrame:
    """
    Converts a dictionary of simulation results into a pandas DataFrame.
    """
    # This is a simple conversion; a real implementation might handle more complex dict structures.
    return pd.DataFrame([simulation_results])
