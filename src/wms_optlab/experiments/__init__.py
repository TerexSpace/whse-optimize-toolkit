"""
Computational Experiments Module for Multi-Objective Integrated Warehouse Optimization.

This module provides the experimental framework for the C&OR paper:
"MOIWOF: A Multi-Objective Integrated Warehouse Optimization Framework 
with Adaptive Decomposition for Joint Slotting-Routing-Batching Problems"
"""

from .moiwof import MOIWOF, MOIWOFConfig, ParetoSolution
from .moiwof_v2 import MOIWOFv2
from .benchmark_generator import BenchmarkInstanceGenerator
from .statistical_analysis import StatisticalAnalyzer
from .visualization import ExperimentVisualizer
from .hypervolume import QualityIndicators, calculate_hypervolume_3d
from .baselines import (
    NSGA2Vanilla,
    MOEAD, 
    SequentialOptimization,
    RandomBaseline,
    ABCHeuristic
)

__all__ = [
    'MOIWOF',
    'MOIWOFConfig', 
    'ParetoSolution',
    'MOIWOFv2',
    'BenchmarkInstanceGenerator',
    'StatisticalAnalyzer',
    'ExperimentVisualizer',
    'QualityIndicators',
    'calculate_hypervolume_3d',
    'NSGA2Vanilla',
    'MOEAD',
    'SequentialOptimization',
    'RandomBaseline',
    'ABCHeuristic'
]
