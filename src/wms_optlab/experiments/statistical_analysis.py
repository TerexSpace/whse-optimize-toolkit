"""
Statistical Analysis Module for Computational Experiments.

Provides rigorous statistical analysis for comparing optimization algorithms:
- Descriptive statistics (mean, std, median, IQR)
- Non-parametric tests (Wilcoxon, Friedman, Kruskal-Wallis)
- Effect size calculations (Vargha-Delaney A12, Cliff's delta)
- Multi-objective quality indicators (Hypervolume, IGD, Spread)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from collections import defaultdict

from .moiwof import ParetoSolution


@dataclass
class StatisticalSummary:
    """Summary statistics for a metric."""
    metric_name: str
    mean: float
    std: float
    median: float
    q1: float
    q3: float
    iqr: float
    min_val: float
    max_val: float
    n: int


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # at alpha=0.05
    effect_size: Optional[float] = None
    effect_interpretation: Optional[str] = None


@dataclass
class QualityIndicators:
    """Multi-objective quality indicators."""
    hypervolume: float
    igd: float  # Inverted Generational Distance
    spread: float
    spacing: float
    pareto_front_size: int


class StatisticalAnalyzer:
    """Performs statistical analysis on experimental results."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def descriptive_stats(self, values: List[float], metric_name: str = "metric") -> StatisticalSummary:
        """Calculate descriptive statistics."""
        arr = np.array(values)
        return StatisticalSummary(
            metric_name=metric_name,
            mean=np.mean(arr),
            std=np.std(arr),
            median=np.median(arr),
            q1=np.percentile(arr, 25),
            q3=np.percentile(arr, 75),
            iqr=np.percentile(arr, 75) - np.percentile(arr, 25),
            min_val=np.min(arr),
            max_val=np.max(arr),
            n=len(arr)
        )
    
    def wilcoxon_test(self, group1: List[float], group2: List[float], 
                      alternative: str = 'two-sided') -> HypothesisTestResult:
        """
        Perform Wilcoxon signed-rank test for paired samples.
        Use when comparing two algorithms on the same instances.
        """
        if len(group1) != len(group2):
            raise ValueError("Groups must have equal length for paired test")
        
        # Check if all differences are zero (would cause error)
        diff = np.array(group1) - np.array(group2)
        if np.all(diff == 0):
            return HypothesisTestResult(
                test_name="Wilcoxon signed-rank",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.5,
                effect_interpretation="negligible"
            )
        
        statistic, p_value = stats.wilcoxon(group1, group2, alternative=alternative)
        
        # Calculate effect size (Vargha-Delaney A12)
        effect_size = self.vargha_delaney_a12(group1, group2)
        effect_interpretation = self._interpret_a12(effect_size)
        
        return HypothesisTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            effect_interpretation=effect_interpretation
        )
    
    def mann_whitney_test(self, group1: List[float], group2: List[float],
                          alternative: str = 'two-sided') -> HypothesisTestResult:
        """
        Perform Mann-Whitney U test for independent samples.
        Use when comparing algorithms on different instances.
        """
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        
        effect_size = self.vargha_delaney_a12(group1, group2)
        effect_interpretation = self._interpret_a12(effect_size)
        
        return HypothesisTestResult(
            test_name="Mann-Whitney U",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            effect_interpretation=effect_interpretation
        )
    
    def friedman_test(self, *groups: List[float]) -> HypothesisTestResult:
        """
        Perform Friedman test for comparing multiple algorithms.
        Use when comparing 3+ algorithms on the same instances.
        """
        statistic, p_value = stats.friedmanchisquare(*groups)
        
        return HypothesisTestResult(
            test_name="Friedman",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha
        )
    
    def kruskal_wallis_test(self, *groups: List[float]) -> HypothesisTestResult:
        """
        Perform Kruskal-Wallis H test for independent samples.
        Use when comparing 3+ algorithms on different instances.
        """
        statistic, p_value = stats.kruskal(*groups)
        
        return HypothesisTestResult(
            test_name="Kruskal-Wallis H",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha
        )
    
    def vargha_delaney_a12(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Vargha-Delaney A12 effect size.
        
        A12 measures the probability that a randomly selected value from group1
        is larger than a randomly selected value from group2.
        
        Interpretation:
        - A12 = 0.5: no difference
        - A12 > 0.5: group1 tends to be larger (worse if minimizing)
        - A12 < 0.5: group1 tends to be smaller (better if minimizing)
        """
        m, n = len(group1), len(group2)
        r1 = sum(1 for x1 in group1 for x2 in group2 if x1 > x2)
        r2 = sum(0.5 for x1 in group1 for x2 in group2 if x1 == x2)
        
        return (r1 + r2) / (m * n)
    
    def cliffs_delta(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cliff's delta effect size.
        Related to A12: delta = 2*A12 - 1
        
        Interpretation:
        - |delta| < 0.147: negligible
        - 0.147 <= |delta| < 0.33: small
        - 0.33 <= |delta| < 0.474: medium
        - |delta| >= 0.474: large
        """
        a12 = self.vargha_delaney_a12(group1, group2)
        return 2 * a12 - 1
    
    def _interpret_a12(self, a12: float) -> str:
        """Interpret A12 effect size."""
        delta = abs(2 * a12 - 1)
        if delta < 0.147:
            return "negligible"
        elif delta < 0.33:
            return "small"
        elif delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def calculate_hypervolume(self, pareto_front: List[ParetoSolution],
                               reference_point: Dict[str, float]) -> float:
        """
        Calculate hypervolume indicator for a Pareto front.
        Higher is better.
        """
        if not pareto_front:
            return 0.0
        
        objectives = list(pareto_front[0].objectives.keys())
        
        # Monte Carlo approximation for 3D case
        n_samples = 50000
        count_dominated = 0
        
        mins = {obj: min(sol.objectives[obj] for sol in pareto_front) for obj in objectives}
        maxs = reference_point
        
        for _ in range(n_samples):
            sample = {
                obj: np.random.uniform(mins[obj], maxs[obj])
                for obj in objectives
            }
            
            for sol in pareto_front:
                if all(sol.objectives[obj] <= sample[obj] for obj in objectives):
                    count_dominated += 1
                    break
        
        # Calculate volume of sampling region
        total_volume = 1.0
        for obj in objectives:
            total_volume *= (maxs[obj] - mins[obj])
        
        return (count_dominated / n_samples) * total_volume
    
    def calculate_igd(self, pareto_front: List[ParetoSolution],
                       true_front: List[Dict[str, float]]) -> float:
        """
        Calculate Inverted Generational Distance (IGD).
        Lower is better.
        """
        if not pareto_front or not true_front:
            return float('inf')
        
        objectives = list(pareto_front[0].objectives.keys())
        
        # For each point in true front, find minimum distance to obtained front
        distances = []
        for true_point in true_front:
            min_dist = float('inf')
            for sol in pareto_front:
                dist = np.sqrt(sum(
                    (sol.objectives[obj] - true_point[obj]) ** 2
                    for obj in objectives
                ))
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        return np.mean(distances)
    
    def calculate_spread(self, pareto_front: List[ParetoSolution]) -> float:
        """
        Calculate spread/diversity of Pareto front.
        Measures how well-distributed solutions are.
        """
        if len(pareto_front) < 3:
            return 0.0
        
        objectives = list(pareto_front[0].objectives.keys())
        
        # Calculate consecutive distances
        distances = []
        for obj in objectives:
            sorted_sols = sorted(pareto_front, key=lambda s: s.objectives[obj])
            for i in range(len(sorted_sols) - 1):
                dist = abs(sorted_sols[i+1].objectives[obj] - sorted_sols[i].objectives[obj])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        d_mean = np.mean(distances)
        spread = np.sum(np.abs(np.array(distances) - d_mean)) / len(distances)
        
        return spread
    
    def calculate_spacing(self, pareto_front: List[ParetoSolution]) -> float:
        """
        Calculate spacing metric (uniformity of distribution).
        Lower is better.
        """
        if len(pareto_front) < 2:
            return 0.0
        
        objectives = list(pareto_front[0].objectives.keys())
        
        # Calculate minimum distance to nearest neighbor for each solution
        min_distances = []
        for i, sol1 in enumerate(pareto_front):
            min_dist = float('inf')
            for j, sol2 in enumerate(pareto_front):
                if i != j:
                    dist = sum(
                        abs(sol1.objectives[obj] - sol2.objectives[obj])
                        for obj in objectives
                    )
                    min_dist = min(min_dist, dist)
            if min_dist < float('inf'):
                min_distances.append(min_dist)
        
        if not min_distances:
            return 0.0
        
        d_mean = np.mean(min_distances)
        spacing = np.sqrt(np.mean((np.array(min_distances) - d_mean) ** 2))
        
        return spacing
    
    def evaluate_pareto_front(self, pareto_front: List[ParetoSolution],
                               reference_point: Optional[Dict[str, float]] = None,
                               true_front: Optional[List[Dict[str, float]]] = None) -> QualityIndicators:
        """Calculate all quality indicators for a Pareto front."""
        if not pareto_front:
            return QualityIndicators(
                hypervolume=0.0,
                igd=float('inf'),
                spread=0.0,
                spacing=0.0,
                pareto_front_size=0
            )
        
        # Default reference point if not provided
        if reference_point is None:
            objectives = list(pareto_front[0].objectives.keys())
            reference_point = {
                obj: max(sol.objectives[obj] for sol in pareto_front) * 1.1
                for obj in objectives
            }
        
        # Calculate IGD if true front provided, else use 0
        igd = 0.0
        if true_front:
            igd = self.calculate_igd(pareto_front, true_front)
        
        return QualityIndicators(
            hypervolume=self.calculate_hypervolume(pareto_front, reference_point),
            igd=igd,
            spread=self.calculate_spread(pareto_front),
            spacing=self.calculate_spacing(pareto_front),
            pareto_front_size=len(pareto_front)
        )
    
    def compare_algorithms(self, 
                           results: Dict[str, List[Dict[str, float]]],
                           metric: str) -> Dict[str, Any]:
        """
        Comprehensive comparison of multiple algorithms.
        
        Args:
            results: Dictionary mapping algorithm name to list of metric results
            metric: The metric to compare
            
        Returns:
            Dictionary with statistical comparisons
        """
        algorithms = list(results.keys())
        
        # Extract metric values for each algorithm
        metric_values = {
            alg: [r[metric] for r in results[alg] if metric in r]
            for alg in algorithms
        }
        
        # Descriptive statistics
        descriptive = {
            alg: self.descriptive_stats(vals, f"{alg}_{metric}")
            for alg, vals in metric_values.items()
            if vals
        }
        
        # Pairwise comparisons
        pairwise = {}
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                if metric_values[alg1] and metric_values[alg2]:
                    key = f"{alg1}_vs_{alg2}"
                    if len(metric_values[alg1]) == len(metric_values[alg2]):
                        pairwise[key] = self.wilcoxon_test(
                            metric_values[alg1], metric_values[alg2]
                        )
                    else:
                        pairwise[key] = self.mann_whitney_test(
                            metric_values[alg1], metric_values[alg2]
                        )
        
        # Overall comparison (if 3+ algorithms)
        overall = None
        if len(algorithms) >= 3:
            groups = [metric_values[alg] for alg in algorithms if metric_values[alg]]
            if all(len(g) == len(groups[0]) for g in groups):
                overall = self.friedman_test(*groups)
            else:
                overall = self.kruskal_wallis_test(*groups)
        
        # Rank algorithms by mean performance
        ranking = sorted(
            [(alg, descriptive[alg].mean) for alg in descriptive],
            key=lambda x: x[1]
        )
        
        return {
            'metric': metric,
            'descriptive': descriptive,
            'pairwise_tests': pairwise,
            'overall_test': overall,
            'ranking': ranking
        }
    
    def generate_latex_table(self, comparison_results: Dict[str, Any]) -> str:
        """Generate LaTeX table from comparison results."""
        descriptive = comparison_results['descriptive']
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Statistical comparison for {comparison_results['metric']}}}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Algorithm & Mean & Std & Median & Q1 & Q3 & n \\\\",
            "\\midrule"
        ]
        
        for alg, stats in descriptive.items():
            lines.append(
                f"{alg} & {stats.mean:.4f} & {stats.std:.4f} & {stats.median:.4f} & "
                f"{stats.q1:.4f} & {stats.q3:.4f} & {stats.n} \\\\"
            )
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return '\n'.join(lines)
