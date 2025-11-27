from typing import List, Dict, Callable
from collections import Counter
from ..data.models import SKU, Location, Order
from ..layout.geometry import Point

def assign_by_abc_popularity(
    skus: List[SKU],
    locations: List[Location],
    orders: List[Order],
    distance_metric: Callable[[Point, Point], float],
    depot_location: Point = (0, 0, 0)
) -> Dict[str, str]:
    """
    Assigns SKUs to locations based on popularity (ABC analysis).
    The most frequently ordered SKUs are placed in the locations closest to the depot.

    Args:
        skus: List of SKUs to be slotted.
        locations: List of available storage locations.
        orders: List of historical or simulated orders.
        distance_metric: A function to calculate distance (e.g., manhattan_distance).
        depot_location: The coordinates of the depot or packing station.

    Returns:
        A dictionary mapping SKU IDs to Location IDs.
    """
    # 1. Calculate SKU popularity from orders
    sku_demand = Counter()
    for order in orders:
        for line in order.order_lines:
            sku_demand[line.sku.sku_id] += line.quantity
    
    # Sort SKUs by popularity (descending) and SKU ID to keep ordering deterministic.
    sorted_skus = sorted(
        skus,
        key=lambda s: (-sku_demand.get(s.sku_id, 0), s.sku_id)
    )

    # 2. Sort locations by distance from depot (ascending)
    sorted_locations = sorted(
        locations,
        key=lambda loc: distance_metric(loc.coordinates, depot_location)
    )

    # 3. Assign most popular SKUs to closest locations
    slotting_plan = {}
    num_locations = len(sorted_locations)
    for i, sku in enumerate(sorted_skus):
        if i < num_locations:
            slotting_plan[sku.sku_id] = sorted_locations[i].loc_id
        else:
            break  # No more locations left

    return slotting_plan
