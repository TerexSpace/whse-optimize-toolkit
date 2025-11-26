from typing import Tuple

Point = Tuple[float, float, float]

def manhattan_distance(p1: Point, p2: Point) -> float:
    """
    Calculates the Manhattan distance (L1 norm) between two points in 3D space.
    This is a common distance metric for grid-like warehouse layouts.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

def euclidean_distance(p1: Point, p2: Point) -> float:
    """
    Calculates the Euclidean distance (L2 norm) between two points in 3D space.
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5
