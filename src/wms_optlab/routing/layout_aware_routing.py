"""
Layout-Aware Routing Policies for Different Warehouse Configurations.

This module implements specialized routing algorithms optimized for different
warehouse layout types:
- Parallel-Aisle: Standard S-shape/serpentine traversal
- Fishbone: Diagonal cross-aisle aware routing
- Flying-V: V-shaped cross-aisle routing
- Adaptive: Automatic layout detection and policy selection

The key insight from reviewer feedback is that S-shape routing performs poorly
on fishbone layouts because it ignores the diagonal aisle structure. This module
addresses that limitation.

References:
- Ratliff & Rosenthal (1983): Optimal routing in rectangular warehouses
- Gue & Meller (2009): Fishbone aisle design
- Öztürkoğlu et al. (2012): Flying-V aisle design
"""

from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import math
import networkx as nx
import numpy as np
from dataclasses import dataclass

from ..data.models import Location
from ..layout.geometry import manhattan_distance, euclidean_distance, Point


class LayoutType(Enum):
    """Warehouse layout types."""
    PARALLEL_AISLE = "parallel_aisle"
    FISHBONE = "fishbone"
    FLYING_V = "flying_v"
    SINGLE_AISLE = "single_aisle"
    UNKNOWN = "unknown"


@dataclass
class LayoutCharacteristics:
    """Detected characteristics of a warehouse layout."""
    layout_type: LayoutType
    num_aisles: int
    avg_aisle_length: float
    depot_position: Tuple[float, float, float]
    has_cross_aisle: bool
    cross_aisle_angle: float  # 0 for perpendicular, positive for diagonal
    confidence: float  # Detection confidence 0-1


def detect_layout_type(locations: List[Location]) -> LayoutCharacteristics:
    """
    Automatically detect warehouse layout type from location coordinates.
    
    Detection strategy:
    1. Identify depot (location_type='depot')
    2. Analyze x-coordinate distribution for aisle structure
    3. Check for diagonal patterns (fishbone/flying-v indicators)
    4. Return detected type with confidence score
    
    Args:
        locations: List of warehouse locations
        
    Returns:
        LayoutCharacteristics with detected layout info
    """
    storage_locs = [loc for loc in locations if loc.location_type == 'storage']
    depot = next((loc for loc in locations if loc.location_type == 'depot'), None)
    depot_pos = depot.coordinates if depot else (0.0, 0.0, 0.0)
    
    if len(storage_locs) < 5:
        return LayoutCharacteristics(
            layout_type=LayoutType.UNKNOWN,
            num_aisles=1,
            avg_aisle_length=0.0,
            depot_position=depot_pos,
            has_cross_aisle=False,
            cross_aisle_angle=0.0,
            confidence=0.0
        )
    
    # Extract coordinates
    x_coords = [loc.coordinates[0] for loc in storage_locs]
    y_coords = [loc.coordinates[1] for loc in storage_locs]
    
    # Analyze x-coordinate clustering (aisles)
    x_unique = sorted(set(round(x, 1) for x in x_coords))
    x_groups = _cluster_coordinates(x_coords, threshold=1.5)
    num_aisles = len(x_groups)
    
    # Calculate average aisle length
    y_range = max(y_coords) - min(y_coords) if y_coords else 0
    avg_aisle_length = y_range
    
    # Check for diagonal patterns (fishbone indicator)
    # In fishbone, x and y are correlated due to angled slots
    if len(x_coords) > 10:
        correlation = np.corrcoef(x_coords, y_coords)[0, 1]
        has_diagonal = abs(correlation) > 0.3
    else:
        has_diagonal = False
        correlation = 0.0
    
    # Check for symmetric patterns (flying-v indicator)
    x_centered = [x - np.mean(x_coords) for x in x_coords]
    y_centered = [y - np.mean(y_coords) for y in y_coords]
    
    # Flying-V has V-shaped deviation from center
    v_pattern_score = 0.0
    if len(x_centered) > 20:
        # Check if |x| correlates with y (V-shape)
        abs_x = [abs(x) for x in x_centered]
        v_corr = np.corrcoef(abs_x, y_centered)[0, 1]
        v_pattern_score = abs(v_corr)
    
    # Determine layout type
    if v_pattern_score > 0.4:
        layout_type = LayoutType.FLYING_V
        cross_aisle_angle = 30.0  # Approximate V angle
        confidence = min(0.9, 0.5 + v_pattern_score)
    elif has_diagonal and abs(correlation) > 0.3:
        layout_type = LayoutType.FISHBONE
        cross_aisle_angle = math.degrees(math.atan(abs(correlation)))
        confidence = min(0.9, 0.5 + abs(correlation))
    elif num_aisles >= 2:
        layout_type = LayoutType.PARALLEL_AISLE
        cross_aisle_angle = 0.0
        confidence = 0.85
    else:
        layout_type = LayoutType.SINGLE_AISLE
        cross_aisle_angle = 0.0
        confidence = 0.7
    
    return LayoutCharacteristics(
        layout_type=layout_type,
        num_aisles=num_aisles,
        avg_aisle_length=avg_aisle_length,
        depot_position=depot_pos,
        has_cross_aisle=layout_type in [LayoutType.FISHBONE, LayoutType.FLYING_V],
        cross_aisle_angle=cross_aisle_angle,
        confidence=confidence
    )


def _cluster_coordinates(coords: List[float], threshold: float = 1.5) -> List[List[float]]:
    """Cluster coordinates into groups based on proximity."""
    if not coords:
        return []
    
    sorted_coords = sorted(coords)
    clusters = [[sorted_coords[0]]]
    
    for coord in sorted_coords[1:]:
        if coord - clusters[-1][-1] <= threshold:
            clusters[-1].append(coord)
        else:
            clusters.append([coord])
    
    return clusters


def get_s_shape_route(
    pick_locations: List[Location],
    warehouse_graph: nx.Graph,
    depot_loc: Location
) -> List[str]:
    """
    S-shape (serpentine) routing for parallel-aisle warehouses.
    
    Optimal for standard rectangular warehouses with perpendicular aisles.
    
    Args:
        pick_locations: Locations to visit
        warehouse_graph: Warehouse topology graph
        depot_loc: Start/end depot location
        
    Returns:
        Ordered list of location IDs forming the route
    """
    if not pick_locations:
        return [depot_loc.loc_id, depot_loc.loc_id]

    route = [depot_loc.loc_id]
    
    # Group locations by aisle (x-coordinate)
    aisles: Dict[float, List[Location]] = {}
    for loc in pick_locations:
        aisle_x = round(loc.coordinates[0], 1)
        if aisle_x not in aisles:
            aisles[aisle_x] = []
        aisles[aisle_x].append(loc)

    # Sort aisles by x-coordinate
    sorted_aisle_keys = sorted(aisles.keys())

    # Traverse aisles in S-shape pattern
    for i, aisle_x in enumerate(sorted_aisle_keys):
        aisle_locs = aisles[aisle_x]
        
        # Alternate direction based on aisle index
        direction_ascending = (i % 2 == 0)
        aisle_locs.sort(key=lambda loc: loc.coordinates[1], reverse=not direction_ascending)
        
        route.extend([loc.loc_id for loc in aisle_locs])

    route.append(depot_loc.loc_id)
    return route


def get_fishbone_route(
    pick_locations: List[Location],
    warehouse_graph: nx.Graph,
    depot_loc: Location
) -> List[str]:
    """
    Optimized routing for fishbone warehouse layouts.
    
    Fishbone layouts have:
    - Central cross-aisle running through depot
    - Diagonal aisles branching from center
    - Natural V-shaped traversal patterns
    
    Strategy:
    1. Partition picks into left/right halves relative to depot
    2. Process each half using diagonal-aware traversal
    3. Use cross-aisle for efficient transitions
    
    Args:
        pick_locations: Locations to visit
        warehouse_graph: Warehouse topology graph  
        depot_loc: Start/end depot location
        
    Returns:
        Ordered list of location IDs forming the route
    """
    if not pick_locations:
        return [depot_loc.loc_id, depot_loc.loc_id]
    
    depot_x = depot_loc.coordinates[0]
    depot_y = depot_loc.coordinates[1]
    
    # Partition locations into left and right of depot
    left_locs = [loc for loc in pick_locations if loc.coordinates[0] < depot_x]
    right_locs = [loc for loc in pick_locations if loc.coordinates[0] >= depot_x]
    
    route = [depot_loc.loc_id]
    
    # Process left side first (if any picks there)
    if left_locs:
        # Sort by diagonal distance from depot (accounting for fishbone angle)
        left_sorted = sorted(
            left_locs,
            key=lambda loc: _fishbone_priority(loc, depot_loc, side='left')
        )
        route.extend([loc.loc_id for loc in left_sorted])
    
    # Process right side
    if right_locs:
        # Sort by diagonal distance from depot
        right_sorted = sorted(
            right_locs,
            key=lambda loc: _fishbone_priority(loc, depot_loc, side='right')
        )
        route.extend([loc.loc_id for loc in right_sorted])
    
    route.append(depot_loc.loc_id)
    return route


def _fishbone_priority(loc: Location, depot: Location, side: str) -> float:
    """
    Calculate priority for fishbone routing.
    
    In fishbone layouts, diagonal traversal is more efficient.
    Priority considers:
    - Distance from central cross-aisle
    - Diagonal travel efficiency
    - Natural aisle grouping
    """
    dx = abs(loc.coordinates[0] - depot.coordinates[0])
    dy = loc.coordinates[1] - depot.coordinates[1]
    
    # Fishbone angle factor (typical ~30 degrees)
    angle_factor = 0.5  
    
    # Priority based on y-distance adjusted for diagonal access
    diagonal_distance = dy + dx * angle_factor
    
    # Group by approximate aisle (cluster x-coordinates)
    aisle_group = round(loc.coordinates[0] / 5.0)  # 5-unit aisle spacing
    
    return (aisle_group * 1000) + diagonal_distance


def get_flying_v_route(
    pick_locations: List[Location],
    warehouse_graph: nx.Graph,
    depot_loc: Location
) -> List[str]:
    """
    Optimized routing for Flying-V warehouse layouts.
    
    Flying-V layouts have:
    - V-shaped cross-aisles for faster access
    - Optimal for depot at front center
    - Natural branching traversal
    
    Strategy:
    1. Compute angular position of each pick relative to depot
    2. Sort by angle, then by distance
    3. Traverse in angular sweep pattern
    
    Args:
        pick_locations: Locations to visit
        warehouse_graph: Warehouse topology graph
        depot_loc: Start/end depot location
        
    Returns:
        Ordered list of location IDs forming the route
    """
    if not pick_locations:
        return [depot_loc.loc_id, depot_loc.loc_id]
    
    depot_x, depot_y = depot_loc.coordinates[0], depot_loc.coordinates[1]
    
    # Calculate polar coordinates relative to depot
    def angular_priority(loc: Location) -> Tuple[float, float]:
        dx = loc.coordinates[0] - depot_x
        dy = loc.coordinates[1] - depot_y
        angle = math.atan2(dx, dy)  # Angle from depot
        distance = math.sqrt(dx**2 + dy**2)
        return (angle, distance)
    
    # Sort by angle, then distance (nearest-first within angle band)
    sorted_locs = sorted(pick_locations, key=angular_priority)
    
    route = [depot_loc.loc_id]
    route.extend([loc.loc_id for loc in sorted_locs])
    route.append(depot_loc.loc_id)
    
    return route


def get_nearest_neighbor_route(
    pick_locations: List[Location],
    warehouse_graph: nx.Graph,
    depot_loc: Location
) -> List[str]:
    """
    Nearest neighbor heuristic for general layouts.
    
    Simple greedy approach that works reasonably well for any layout.
    Used as fallback when layout type is unknown.
    
    Args:
        pick_locations: Locations to visit
        warehouse_graph: Warehouse topology graph
        depot_loc: Start/end depot location
        
    Returns:
        Ordered list of location IDs forming the route
    """
    if not pick_locations:
        return [depot_loc.loc_id, depot_loc.loc_id]
    
    route = [depot_loc.loc_id]
    remaining = list(pick_locations)
    current = depot_loc
    
    while remaining:
        # Find nearest unvisited location
        nearest = min(
            remaining,
            key=lambda loc: manhattan_distance(current.coordinates, loc.coordinates)
        )
        route.append(nearest.loc_id)
        remaining.remove(nearest)
        current = nearest
    
    route.append(depot_loc.loc_id)
    return route


def get_largest_gap_route(
    pick_locations: List[Location],
    warehouse_graph: nx.Graph,
    depot_loc: Location
) -> List[str]:
    """
    Largest gap routing heuristic.
    
    Only enters aisle if gap between picks is large enough;
    otherwise uses cross-aisle. Good for sparse picks.
    
    Args:
        pick_locations: Locations to visit
        warehouse_graph: Warehouse topology graph
        depot_loc: Start/end depot location
        
    Returns:
        Ordered list of location IDs forming the route
    """
    if not pick_locations:
        return [depot_loc.loc_id, depot_loc.loc_id]
    
    # Group by aisle
    aisles: Dict[float, List[Location]] = {}
    for loc in pick_locations:
        aisle_x = round(loc.coordinates[0], 1)
        if aisle_x not in aisles:
            aisles[aisle_x] = []
        aisles[aisle_x].append(loc)
    
    route = [depot_loc.loc_id]
    sorted_aisle_keys = sorted(aisles.keys())
    
    for i, aisle_x in enumerate(sorted_aisle_keys):
        aisle_locs = sorted(aisles[aisle_x], key=lambda loc: loc.coordinates[1])
        
        if len(aisle_locs) == 1:
            # Single pick: approach from nearest end
            route.append(aisle_locs[0].loc_id)
        else:
            # Find largest gap in aisle
            gaps = []
            for j in range(len(aisle_locs) - 1):
                gap_size = aisle_locs[j+1].coordinates[1] - aisle_locs[j].coordinates[1]
                gaps.append((gap_size, j))
            
            if gaps:
                largest_gap_size, largest_gap_idx = max(gaps)
                
                # Decide traversal based on gap location
                if largest_gap_size > 10.0:  # Threshold for "large" gap
                    # Enter from both ends, skip the gap
                    bottom_picks = aisle_locs[:largest_gap_idx + 1]
                    top_picks = aisle_locs[largest_gap_idx + 1:]
                    
                    if i % 2 == 0:
                        route.extend([loc.loc_id for loc in bottom_picks])
                        route.extend([loc.loc_id for loc in reversed(top_picks)])
                    else:
                        route.extend([loc.loc_id for loc in reversed(top_picks)])
                        route.extend([loc.loc_id for loc in bottom_picks])
                else:
                    # Traverse entire aisle
                    direction = (i % 2 == 0)
                    if not direction:
                        aisle_locs = list(reversed(aisle_locs))
                    route.extend([loc.loc_id for loc in aisle_locs])
            else:
                route.extend([loc.loc_id for loc in aisle_locs])
    
    route.append(depot_loc.loc_id)
    return route


def get_adaptive_route(
    pick_locations: List[Location],
    warehouse_graph: nx.Graph,
    depot_loc: Location,
    all_locations: Optional[List[Location]] = None
) -> List[str]:
    """
    Adaptive routing that automatically selects the best policy for the layout.
    
    This is the recommended entry point for routing. It:
    1. Detects the warehouse layout type
    2. Selects the appropriate routing policy
    3. Returns an optimized route
    
    Args:
        pick_locations: Locations to visit
        warehouse_graph: Warehouse topology graph
        depot_loc: Start/end depot location
        all_locations: All warehouse locations (for layout detection)
        
    Returns:
        Ordered list of location IDs forming the route
    """
    if not pick_locations:
        return [depot_loc.loc_id, depot_loc.loc_id]
    
    # Detect layout type from all locations (or pick locations if not provided)
    detection_locs = all_locations if all_locations else pick_locations + [depot_loc]
    layout_info = detect_layout_type(detection_locs)
    
    # Select routing policy based on detected layout
    if layout_info.layout_type == LayoutType.FISHBONE:
        return get_fishbone_route(pick_locations, warehouse_graph, depot_loc)
    elif layout_info.layout_type == LayoutType.FLYING_V:
        return get_flying_v_route(pick_locations, warehouse_graph, depot_loc)
    elif layout_info.layout_type == LayoutType.PARALLEL_AISLE:
        return get_s_shape_route(pick_locations, warehouse_graph, depot_loc)
    else:
        # Fallback to nearest neighbor for unknown layouts
        return get_nearest_neighbor_route(pick_locations, warehouse_graph, depot_loc)


def get_route_for_layout(
    pick_locations: List[Location],
    warehouse_graph: nx.Graph,
    depot_loc: Location,
    layout_type: LayoutType
) -> List[str]:
    """
    Get route using a specific layout-optimized policy.
    
    Use this when you know the layout type and want to avoid detection overhead.
    
    Args:
        pick_locations: Locations to visit
        warehouse_graph: Warehouse topology graph
        depot_loc: Start/end depot location
        layout_type: Known layout type
        
    Returns:
        Ordered list of location IDs forming the route
    """
    routing_policies = {
        LayoutType.PARALLEL_AISLE: get_s_shape_route,
        LayoutType.FISHBONE: get_fishbone_route,
        LayoutType.FLYING_V: get_flying_v_route,
        LayoutType.SINGLE_AISLE: get_nearest_neighbor_route,
        LayoutType.UNKNOWN: get_nearest_neighbor_route,
    }
    
    policy = routing_policies.get(layout_type, get_nearest_neighbor_route)
    return policy(pick_locations, warehouse_graph, depot_loc)


def calculate_route_distance(
    route: List[str],
    loc_map: Dict[str, Location]
) -> float:
    """
    Calculate total travel distance for a route.
    
    Args:
        route: Ordered list of location IDs
        loc_map: Mapping from location ID to Location object
        
    Returns:
        Total Manhattan distance traveled
    """
    if len(route) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(route) - 1):
        loc1 = loc_map.get(route[i])
        loc2 = loc_map.get(route[i + 1])
        if loc1 and loc2:
            total += manhattan_distance(loc1.coordinates, loc2.coordinates)
    
    return total


def compare_routing_policies(
    pick_locations: List[Location],
    warehouse_graph: nx.Graph,
    depot_loc: Location,
    loc_map: Dict[str, Location]
) -> Dict[str, Tuple[List[str], float]]:
    """
    Compare all routing policies on the same pick set.
    
    Useful for analysis and debugging.
    
    Args:
        pick_locations: Locations to visit
        warehouse_graph: Warehouse topology graph
        depot_loc: Start/end depot location
        loc_map: Mapping from location ID to Location object
        
    Returns:
        Dict mapping policy name to (route, distance)
    """
    policies = {
        's_shape': get_s_shape_route,
        'fishbone': get_fishbone_route,
        'flying_v': get_flying_v_route,
        'nearest_neighbor': get_nearest_neighbor_route,
        'largest_gap': get_largest_gap_route,
    }
    
    results = {}
    for name, policy in policies.items():
        route = policy(pick_locations, warehouse_graph, depot_loc)
        distance = calculate_route_distance(route, loc_map)
        results[name] = (route, distance)
    
    return results
