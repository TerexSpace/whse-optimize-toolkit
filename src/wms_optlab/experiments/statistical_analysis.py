"""
Statistical Analysis Module for Computational Experiments.

Provides rigorous statistical analysis for comparing optimization algorithms:
- Descriptive statistics (mean, std, median, IQR)
- Non-parametric tests (Wilcoxon, Friedman, Kruskal-Wallis)
- Effect size calculations (Vargha-Delaney A12, Cliff's delta, Cohen's d)
- Multiple comparison corrections (Holm-Bonferroni, Benjamini-Hochberg)
- Multi-objective quality indicators (Hypervolume, IGD, Spread)

This module addresses reviewer concerns about lack of statistical significance testing
by providing comprehensive analysis tools with proper multiple comparison corrections.

References:
- Vargha & Delaney (2000): A12 measure
- Holm (1979): Sequential Bonferroni procedure
- Benjamini & Hochberg (1995): False discovery rate control
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import stats
from collections import defaultdict
import warnings

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
    
    def __str__(self) -> str:
        return f"{self.metric_name}: {self.mean:.4f} ± {self.std:.4f} (n={self.n})"


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # at alpha=0.05
    effect_size: Optional[float] = None
    effect_interpretation: Optional[str] = None
    corrected_p_value: Optional[float] = None  # After multiple comparison correction
    cohens_d: Optional[float] = None  # Effect size in standard deviation units
    
    def __str__(self) -> str:
        sig = "***" if self.p_value < 0.001 else ("**" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else ""))
        return f"{self.test_name}: p={self.p_value:.4f}{sig}, effect={self.effect_interpretation or 'N/A'}"


@dataclass
class MultipleComparisonResult:
    """Results from multiple comparison correction."""
    original_p_values: List[float]
    corrected_p_values: List[float]
    rejected: List[bool]
    method: str
    alpha: float
    comparison_labels: List[str] = field(default_factory=list)


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
            std=np.std(arr, ddof=1),  # Sample std with Bessel's correction
            median=np.median(arr),
            q1=np.percentile(arr, 25),
            q3=np.percentile(arr, 75),
            iqr=np.percentile(arr, 75) - np.percentile(arr, 25),
            min_val=np.min(arr),
            max_val=np.max(arr),
            n=len(arr)
        )
    
    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size for two groups.
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def holm_bonferroni_correction(self, p_values: List[float], 
                                    labels: Optional[List[str]] = None) -> MultipleComparisonResult:
        """
        Apply Holm-Bonferroni correction for multiple comparisons.
        
        This is a step-down procedure that controls family-wise error rate (FWER)
        while being more powerful than standard Bonferroni.
        
        Args:
            p_values: List of p-values from pairwise comparisons
            labels: Optional labels for each comparison
            
        Returns:
            MultipleComparisonResult with corrected p-values and rejection decisions
        """
        n = len(p_values)
        if n == 0:
            return MultipleComparisonResult([], [], [], "holm-bonferroni", self.alpha)
        
        # Sort p-values with original indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        
        corrected = [0.0] * n
        rejected = [False] * n
        
        for rank, (orig_idx, p) in enumerate(indexed):
            # Holm correction: multiply by (n - rank)
            corrected_p = min(1.0, p * (n - rank))
            corrected[orig_idx] = corrected_p
            
            # Reject if corrected p < alpha AND all previous (smaller) p-values were rejected
            if rank == 0:
                rejected[orig_idx] = corrected_p < self.alpha
            else:
                # Can only reject if previous hypotheses were rejected
                prev_orig_idx = indexed[rank - 1][0]
                rejected[orig_idx] = rejected[prev_orig_idx] and corrected_p < self.alpha
        
        return MultipleComparisonResult(
            original_p_values=p_values,
            corrected_p_values=corrected,
            rejected=rejected,
            method="holm-bonferroni",
            alpha=self.alpha,
            comparison_labels=labels or [f"comparison_{i}" for i in range(n)]
        )
    
    def benjamini_hochberg_correction(self, p_values: List[float],
                                       labels: Optional[List[str]] = None) -> MultipleComparisonResult:
        """
        Apply Benjamini-Hochberg correction for multiple comparisons.
        
        Controls false discovery rate (FDR), less conservative than Holm-Bonferroni.
        
        Args:
            p_values: List of p-values from pairwise comparisons
            labels: Optional labels for each comparison
            
        Returns:
            MultipleComparisonResult with corrected p-values and rejection decisions
        """
        n = len(p_values)
        if n == 0:
            return MultipleComparisonResult([], [], [], "benjamini-hochberg", self.alpha)
        
        # Sort p-values with original indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        
        corrected = [0.0] * n
        rejected = [False] * n
        
        # Calculate adjusted p-values
        prev_corrected = 1.0
        for rank in range(n - 1, -1, -1):
            orig_idx, p = indexed[rank]
            # BH correction
            corrected_p = min(prev_corrected, p * n / (rank + 1))
            corrected[orig_idx] = corrected_p
            prev_corrected = corrected_p
            rejected[orig_idx] = corrected_p < self.alpha
        
        return MultipleComparisonResult(
            original_p_values=p_values,
            corrected_p_values=corrected,
            rejected=rejected,
            method="benjamini-hochberg",
            alpha=self.alpha,
            comparison_labels=labels or [f"comparison_{i}" for i in range(n)]
        )
    
    def wilcoxon_test(self, group1: List[float], group2: List[float], 
                      alternative: str = 'two-sided') -> HypothesisTestResult:
        """
        Perform Wilcoxon signed-rank test for paired samples.
        Use when comparing two algorithms on the same instances.
        
        Args:
            group1: Results from algorithm 1 (one value per instance)
            group2: Results from algorithm 2 (one value per instance)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            HypothesisTestResult with test statistics and effect sizes
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
                effect_interpretation="negligible",
                cohens_d=0.0
            )
        
        # Handle case with too few non-zero differences
        non_zero_diff = diff[diff != 0]
        if len(non_zero_diff) < 5:
            warnings.warn(f"Only {len(non_zero_diff)} non-zero differences; results may be unreliable")
        
        try:
            statistic, p_value = stats.wilcoxon(group1, group2, alternative=alternative)
        except ValueError as e:
            # Fallback if Wilcoxon fails
            return HypothesisTestResult(
                test_name="Wilcoxon signed-rank",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.5,
                effect_interpretation="negligible (test failed)",
                cohens_d=0.0
            )
        
        # Calculate effect sizes
        effect_size = self.vargha_delaney_a12(group1, group2)
        effect_interpretation = self._interpret_a12(effect_size)
        cohens_d_val = self.cohens_d(group1, group2)
        
        return HypothesisTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            effect_interpretation=effect_interpretation,
            cohens_d=cohens_d_val
        )
        
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
        
        Args:
            group1: Results from algorithm 1
            group2: Results from algorithm 2
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            HypothesisTestResult with test statistics and effect sizes
        """
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        
        effect_size = self.vargha_delaney_a12(group1, group2)
        effect_interpretation = self._interpret_a12(effect_size)
        cohens_d_val = self.cohens_d(group1, group2)
        
        return HypothesisTestResult(
            test_name="Mann-Whitney U",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            effect_interpretation=effect_interpretation,
            cohens_d=cohens_d_val
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
    
    def nemenyi_post_hoc(self, *groups: List[float], 
                         algorithm_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform Nemenyi post-hoc test after Friedman test.
        
        The Nemenyi test is used for pairwise comparisons between algorithms
        when the Friedman test indicates significant differences. It controls
        the family-wise error rate.
        
        Args:
            *groups: Variable number of groups (algorithm results)
            algorithm_names: Optional names for each algorithm
            
        Returns:
            Dictionary with critical difference, average ranks, and significant pairs
        """
        k = len(groups)  # Number of algorithms
        if k < 2:
            return {'error': 'Need at least 2 groups'}
        
        n = len(groups[0])  # Number of instances
        if not all(len(g) == n for g in groups):
            return {'error': 'All groups must have same length'}
        
        if algorithm_names is None:
            algorithm_names = [f"Alg_{i+1}" for i in range(k)]
        
        # Calculate average ranks
        data = np.array(groups).T  # Shape: (n_instances, k_algorithms)
        ranks = np.zeros_like(data, dtype=float)
        
        for i in range(n):
            ranks[i] = stats.rankdata(data[i])
        
        avg_ranks = np.mean(ranks, axis=0)
        
        # Critical difference at alpha=0.05 (tabulated Nemenyi q-values)
        # q_alpha for k groups (approximation for common values)
        q_values = {
            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
            7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
        }
        q_alpha = q_values.get(k, 2.576)  # Default to approximate value
        
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
        
        # Identify significantly different pairs
        significant_pairs = []
        all_comparisons = []
        
        for i in range(k):
            for j in range(i + 1, k):
                diff = abs(avg_ranks[i] - avg_ranks[j])
                is_significant = diff > cd
                all_comparisons.append({
                    'pair': (algorithm_names[i], algorithm_names[j]),
                    'rank_diff': diff,
                    'critical_diff': cd,
                    'significant': is_significant
                })
                if is_significant:
                    significant_pairs.append((algorithm_names[i], algorithm_names[j]))
        
        return {
            'critical_difference': cd,
            'average_ranks': dict(zip(algorithm_names, avg_ranks)),
            'rank_order': sorted(zip(algorithm_names, avg_ranks), key=lambda x: x[1]),
            'significant_pairs': significant_pairs,
            'all_comparisons': all_comparisons,
            'n_instances': n,
            'n_algorithms': k
        }
    
    def epsilon_sensitivity_analysis(self, 
                                     results: Dict[str, List[float]],
                                     epsilon_values: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Perform epsilon-dominance sensitivity analysis.
        
        Analyzes how robust algorithm rankings are to small performance differences.
        This helps determine if performance differences are practically significant
        beyond statistical significance.
        
        Args:
            results: Dictionary mapping algorithm name to list of metric values
            epsilon_values: List of epsilon thresholds to test (default: [0.01, 0.05, 0.1])
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        if epsilon_values is None:
            epsilon_values = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        
        algorithms = list(results.keys())
        means = {alg: np.mean(results[alg]) for alg in algorithms}
        best_mean = min(means.values())
        
        sensitivity_results = []
        
        for eps in epsilon_values:
            threshold = best_mean * (1 + eps)
            within_epsilon = [alg for alg in algorithms if means[alg] <= threshold]
            
            # Calculate relative improvements needed for non-winning algorithms
            improvements_needed = {}
            for alg in algorithms:
                if alg not in within_epsilon:
                    improvement = (means[alg] - threshold) / means[alg] * 100
                    improvements_needed[alg] = improvement
            
            sensitivity_results.append({
                'epsilon': eps,
                'threshold': threshold,
                'winners': within_epsilon,
                'n_winners': len(within_epsilon),
                'improvements_needed': improvements_needed
            })
        
        # Find stability point (epsilon where ranking becomes stable)
        stability_point = None
        for i in range(len(sensitivity_results) - 1):
            if sensitivity_results[i]['n_winners'] == sensitivity_results[i+1]['n_winners']:
                stability_point = sensitivity_results[i]['epsilon']
                break
        
        return {
            'algorithm_means': means,
            'best_algorithm': min(means, key=means.get),
            'best_mean': best_mean,
            'sensitivity_by_epsilon': sensitivity_results,
            'stability_point': stability_point,
            'practical_significance': self._assess_practical_significance(results)
        }
    
    def _assess_practical_significance(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Assess practical significance of differences between algorithms."""
        algorithms = list(results.keys())
        means = {alg: np.mean(results[alg]) for alg in algorithms}
        
        # Sort by mean
        sorted_algs = sorted(algorithms, key=lambda a: means[a])
        
        # Calculate percentage differences between adjacent algorithms
        pct_diffs = []
        for i in range(len(sorted_algs) - 1):
            alg1, alg2 = sorted_algs[i], sorted_algs[i+1]
            pct_diff = (means[alg2] - means[alg1]) / means[alg1] * 100
            pct_diffs.append({
                'better': alg1,
                'worse': alg2,
                'percentage_difference': pct_diff
            })
        
        return {
            'ranking': sorted_algs,
            'adjacent_differences': pct_diffs,
            'total_range_pct': (max(means.values()) - min(means.values())) / min(means.values()) * 100
        }
    
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
