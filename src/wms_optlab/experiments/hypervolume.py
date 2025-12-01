"""
Hypervolume Indicator Computation for Multi-Objective Optimization.

Implements exact and approximate hypervolume calculation for 2D and 3D objectives.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .moiwof import ParetoSolution, ObjectiveType


def calculate_hypervolume_2d(pareto_front: List[ParetoSolution],
                              obj1: str, obj2: str,
                              reference_point: Optional[Tuple[float, float]] = None) -> float:
    """
    Calculate exact hypervolume for 2D objective space.
    
    Args:
        pareto_front: List of Pareto solutions
        obj1, obj2: Objective names
        reference_point: Reference point (worst values). Auto-computed if None.
    
    Returns:
        Hypervolume value
    """
    if not pareto_front:
        return 0.0
    
    # Extract objective values
    points = [(sol.objectives[obj1], sol.objectives[obj2]) for sol in pareto_front]
    
    # Auto-compute reference point if not provided
    if reference_point is None:
        ref1 = max(p[0] for p in points) * 1.1
        ref2 = max(p[1] for p in points) * 1.1
        reference_point = (ref1, ref2)
    
    # Sort by first objective
    points.sort(key=lambda p: p[0])
    
    # Calculate hypervolume using sweep line algorithm
    hv = 0.0
    prev_y = reference_point[1]
    
    for point in points:
        if point[0] < reference_point[0] and point[1] < reference_point[1]:
            width = reference_point[0] - point[0]
            height = prev_y - point[1]
            if height > 0:
                hv += width * height
            prev_y = min(prev_y, point[1])
    
    return hv


def calculate_hypervolume_3d(pareto_front: List[ParetoSolution],
                              reference_point: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate hypervolume for 3D objective space using WFG algorithm variant.
    
    Args:
        pareto_front: List of Pareto solutions
        reference_point: Reference point dict. Auto-computed if None.
    
    Returns:
        Hypervolume value
    """
    if not pareto_front:
        return 0.0
    
    objectives = [
        ObjectiveType.TRAVEL_DISTANCE.value,
        ObjectiveType.THROUGHPUT_TIME.value,
        ObjectiveType.WORKLOAD_BALANCE.value
    ]
    
    # Extract points as numpy array
    points = np.array([
        [sol.objectives[obj] for obj in objectives]
        for sol in pareto_front
    ])
    
    # Auto-compute reference point
    if reference_point is None:
        ref = points.max(axis=0) * 1.1
    else:
        ref = np.array([reference_point[obj] for obj in objectives])
    
    # Filter points that are dominated by reference
    valid_mask = np.all(points < ref, axis=1)
    points = points[valid_mask]
    
    if len(points) == 0:
        return 0.0
    
    # Use inclusion-exclusion principle for 3D
    return _hv3d_wfg(points, ref)


def _hv3d_wfg(points: np.ndarray, ref: np.ndarray) -> float:
    """
    WFG algorithm for 3D hypervolume.
    O(n log n) for 3D case.
    """
    n = len(points)
    if n == 0:
        return 0.0
    
    # Sort by first objective (descending)
    sorted_idx = np.argsort(-points[:, 0])
    points = points[sorted_idx]
    
    # Initialize
    hv = 0.0
    front_2d = []  # 2D front for projection
    
    for i in range(n):
        point = points[i]
        
        # Calculate contribution in 2D projection
        contribution_2d = _calculate_2d_contribution(
            front_2d, point[1], point[2], ref[1], ref[2]
        )
        
        # Multiply by depth in first dimension
        if i < n - 1:
            depth = points[i + 1, 0] - point[0]
        else:
            depth = ref[0] - point[0]
        
        hv += contribution_2d * depth
        
        # Update 2D front
        front_2d = _update_2d_front(front_2d, (point[1], point[2]))
    
    return hv


def _calculate_2d_contribution(front: List[Tuple[float, float]], 
                                y: float, z: float,
                                ref_y: float, ref_z: float) -> float:
    """Calculate 2D hypervolume contribution of a new point."""
    if not front:
        return (ref_y - y) * (ref_z - z)
    
    # Add new point and calculate full 2D hypervolume
    new_front = front + [(y, z)]
    return _hv2d_exact(new_front, ref_y, ref_z)


def _hv2d_exact(points: List[Tuple[float, float]], ref_y: float, ref_z: float) -> float:
    """Exact 2D hypervolume calculation."""
    if not points:
        return 0.0
    
    # Sort by y coordinate
    points = sorted(points, key=lambda p: p[0])
    
    hv = 0.0
    prev_z = ref_z
    
    for y, z in points:
        if y < ref_y and z < ref_z:
            width = ref_y - y
            height = prev_z - z
            if height > 0:
                hv += width * height
            prev_z = min(prev_z, z)
    
    return hv


def _update_2d_front(front: List[Tuple[float, float]], 
                      new_point: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Update 2D front with new point, removing dominated points."""
    y, z = new_point
    
    # Remove points dominated by new point
    new_front = [(py, pz) for py, pz in front if not (y <= py and z <= pz)]
    
    # Check if new point is dominated
    dominated = any(py <= y and pz <= z for py, pz in front)
    
    if not dominated:
        new_front.append(new_point)
    
    return new_front


def calculate_hypervolume_monte_carlo(pareto_front: List[ParetoSolution],
                                       n_samples: int = 100000,
                                       reference_point: Optional[Dict[str, float]] = None) -> float:
    """
    Monte Carlo approximation of hypervolume.
    Useful for higher dimensions or quick estimates.
    
    Args:
        pareto_front: List of Pareto solutions
        n_samples: Number of Monte Carlo samples
        reference_point: Reference point dict. Auto-computed if None.
    
    Returns:
        Approximate hypervolume value
    """
    if not pareto_front:
        return 0.0
    
    objectives = list(pareto_front[0].objectives.keys())
    
    # Extract bounds
    points = np.array([
        [sol.objectives[obj] for obj in objectives]
        for sol in pareto_front
    ])
    
    mins = points.min(axis=0)
    
    if reference_point is None:
        maxs = points.max(axis=0) * 1.1
    else:
        maxs = np.array([reference_point[obj] for obj in objectives])
    
    # Monte Carlo sampling
    samples = np.random.uniform(mins, maxs, size=(n_samples, len(objectives)))
    
    # Count dominated samples
    dominated_count = 0
    for sample in samples:
        for point in points:
            if np.all(point <= sample):
                dominated_count += 1
                break
    
    # Calculate volume
    total_volume = np.prod(maxs - mins)
    return (dominated_count / n_samples) * total_volume


def calculate_igd(pareto_front: List[ParetoSolution],
                  true_front: List[ParetoSolution]) -> float:
    """
    Calculate Inverted Generational Distance (IGD).
    
    Measures how close the obtained front is to the true Pareto front.
    Lower values indicate better convergence.
    
    Args:
        pareto_front: Obtained Pareto front
        true_front: True/reference Pareto front
    
    Returns:
        IGD value
    """
    if not pareto_front or not true_front:
        return float('inf')
    
    objectives = list(pareto_front[0].objectives.keys())
    
    obtained = np.array([
        [sol.objectives[obj] for obj in objectives]
        for sol in pareto_front
    ])
    
    reference = np.array([
        [sol.objectives[obj] for obj in objectives]
        for sol in true_front
    ])
    
    # Normalize
    all_points = np.vstack([obtained, reference])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    ranges = maxs - mins + 1e-10
    
    obtained_norm = (obtained - mins) / ranges
    reference_norm = (reference - mins) / ranges
    
    # Calculate minimum distances
    distances = []
    for ref_point in reference_norm:
        min_dist = min(np.linalg.norm(ref_point - obt_point) 
                       for obt_point in obtained_norm)
        distances.append(min_dist)
    
    return np.mean(distances)


def calculate_spread(pareto_front: List[ParetoSolution]) -> float:
    """
    Calculate spread/diversity metric.
    
    Measures how well-distributed the solutions are along the Pareto front.
    Higher values indicate better spread.
    
    Args:
        pareto_front: Pareto front solutions
    
    Returns:
        Spread metric value
    """
    if len(pareto_front) < 2:
        return 0.0
    
    objectives = list(pareto_front[0].objectives.keys())
    
    points = np.array([
        [sol.objectives[obj] for obj in objectives]
        for sol in pareto_front
    ])
    
    # Normalize
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins + 1e-10
    points_norm = (points - mins) / ranges
    
    # Calculate nearest neighbor distances
    nn_distances = []
    for i, point in enumerate(points_norm):
        distances = [np.linalg.norm(point - points_norm[j]) 
                    for j in range(len(points_norm)) if i != j]
        nn_distances.append(min(distances))
    
    # Return coefficient of variation (lower = more uniform spread)
    mean_dist = np.mean(nn_distances)
    std_dist = np.std(nn_distances)
    
    if mean_dist == 0:
        return 0.0
    
    # Return inverted CV so higher = better spread
    return 1.0 / (1.0 + std_dist / mean_dist)


class QualityIndicators:
    """Compute and store multiple quality indicators."""
    
    def __init__(self, pareto_front: List[ParetoSolution],
                 reference_front: Optional[List[ParetoSolution]] = None,
                 reference_point: Optional[Dict[str, float]] = None):
        self.pareto_front = pareto_front
        self.reference_front = reference_front
        self.reference_point = reference_point
        
        self._compute_indicators()
    
    def _compute_indicators(self):
        """Compute all quality indicators."""
        self.hypervolume = calculate_hypervolume_3d(
            self.pareto_front, self.reference_point
        )
        
        self.spread = calculate_spread(self.pareto_front)
        
        self.pareto_size = len(self.pareto_front)
        
        if self.reference_front:
            self.igd = calculate_igd(self.pareto_front, self.reference_front)
        else:
            self.igd = None
        
        # Extreme solutions
        objectives = [
            ObjectiveType.TRAVEL_DISTANCE.value,
            ObjectiveType.THROUGHPUT_TIME.value,
            ObjectiveType.WORKLOAD_BALANCE.value
        ]
        
        self.extremes = {}
        for obj in objectives:
            if self.pareto_front:
                best = min(self.pareto_front, key=lambda s: s.objectives[obj])
                self.extremes[obj] = best.objectives[obj]
            else:
                self.extremes[obj] = float('inf')
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'hypervolume': self.hypervolume,
            'spread': self.spread,
            'pareto_size': self.pareto_size,
            'igd': self.igd,
            'best_travel_distance': self.extremes.get(ObjectiveType.TRAVEL_DISTANCE.value),
            'best_throughput_time': self.extremes.get(ObjectiveType.THROUGHPUT_TIME.value),
            'best_workload_balance': self.extremes.get(ObjectiveType.WORKLOAD_BALANCE.value)
        }
    
    def __repr__(self):
        return (f"QualityIndicators(HV={self.hypervolume:.2f}, "
                f"Spread={self.spread:.4f}, Size={self.pareto_size})")
