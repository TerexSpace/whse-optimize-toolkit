"""
Visualization Module for Computational Experiments.

Generates publication-quality figures for:
- Pareto front visualizations (2D and 3D)
- Convergence plots
- Box plots for algorithm comparison
- Sensitivity analysis heatmaps
- Adaptive weight evolution
- Warehouse layout with routes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional, Any
import os

from .moiwof import ParetoSolution, ObjectiveType
from .statistical_analysis import StatisticalSummary


class ExperimentVisualizer:
    """Creates publication-quality visualizations for experiments."""
    
    # IEEE/Elsevier style settings
    STYLE_CONFIG = {
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': (6.5, 4.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    }
    
    # Color palette for algorithms
    COLORS = {
        'MOIWOF': '#1f77b4',      # Blue
        'NSGA-II': '#ff7f0e',     # Orange  
        'SPEA2': '#2ca02c',       # Green
        'MOEA/D': '#d62728',      # Red
        'ABC': '#9467bd',         # Purple
        'Random': '#8c564b',      # Brown
        'Heuristic': '#7f7f7f',   # Gray
    }
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update(self.STYLE_CONFIG)
    
    def plot_pareto_front_2d(self, 
                              pareto_fronts: Dict[str, List[ParetoSolution]],
                              obj_x: str,
                              obj_y: str,
                              filename: str = "pareto_front_2d.pdf",
                              title: str = None) -> str:
        """
        Plot 2D Pareto front comparison.
        
        Args:
            pareto_fronts: Dictionary mapping algorithm name to Pareto front
            obj_x: Objective for x-axis
            obj_y: Objective for y-axis
            filename: Output filename
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(6.5, 5))
        
        for alg_name, front in pareto_fronts.items():
            if not front:
                continue
            
            x_vals = [sol.objectives[obj_x] for sol in front]
            y_vals = [sol.objectives[obj_y] for sol in front]
            
            color = self.COLORS.get(alg_name, '#000000')
            
            # Sort by x for connected line
            sorted_pairs = sorted(zip(x_vals, y_vals))
            x_sorted, y_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
            
            ax.scatter(x_sorted, y_sorted, c=color, label=alg_name, 
                      alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
            ax.plot(x_sorted, y_sorted, c=color, alpha=0.3, linestyle='--')
        
        ax.set_xlabel(self._format_objective_label(obj_x))
        ax.set_ylabel(self._format_objective_label(obj_y))
        
        if title:
            ax.set_title(title)
        
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Add arrow indicating optimization direction
        ax.annotate('', xy=(0.05, 0.05), xytext=(0.15, 0.15),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        ax.text(0.08, 0.08, 'Better', transform=ax.transAxes, 
               fontsize=8, color='gray')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def plot_pareto_front_3d(self,
                              pareto_fronts: Dict[str, List[ParetoSolution]],
                              filename: str = "pareto_front_3d.pdf",
                              title: str = None) -> str:
        """Plot 3D Pareto front with all three objectives."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        obj_names = [
            ObjectiveType.TRAVEL_DISTANCE.value,
            ObjectiveType.THROUGHPUT_TIME.value,
            ObjectiveType.WORKLOAD_BALANCE.value
        ]
        
        for alg_name, front in pareto_fronts.items():
            if not front:
                continue
            
            x_vals = [sol.objectives[obj_names[0]] for sol in front]
            y_vals = [sol.objectives[obj_names[1]] for sol in front]
            z_vals = [sol.objectives[obj_names[2]] for sol in front]
            
            color = self.COLORS.get(alg_name, '#000000')
            ax.scatter(x_vals, y_vals, z_vals, c=color, label=alg_name,
                      alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel(self._format_objective_label(obj_names[0]))
        ax.set_ylabel(self._format_objective_label(obj_names[1]))
        ax.set_zlabel(self._format_objective_label(obj_names[2]))
        
        if title:
            ax.set_title(title)
        
        ax.legend(loc='upper left', framealpha=0.9)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def plot_convergence(self,
                          histories: Dict[str, List[Dict[str, Any]]],
                          metric: str,
                          filename: str = "convergence.pdf",
                          title: str = None,
                          log_scale: bool = False) -> str:
        """
        Plot convergence curves for multiple algorithms.
        
        Args:
            histories: Dictionary mapping algorithm name to generation history
            metric: The metric to plot (e.g., 'best_travel_distance')
            filename: Output filename
            title: Plot title
            log_scale: Whether to use log scale for y-axis
        """
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        
        for alg_name, history in histories.items():
            if not history:
                continue
            
            generations = [h['generation'] for h in history]
            values = [h.get(metric, 0) for h in history]
            
            color = self.COLORS.get(alg_name, '#000000')
            ax.plot(generations, values, color=color, label=alg_name, linewidth=1.5)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel(self._format_objective_label(metric))
        
        if log_scale:
            ax.set_yscale('log')
        
        if title:
            ax.set_title(title)
        
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def plot_boxplots(self,
                       results: Dict[str, List[float]],
                       metric: str,
                       filename: str = "boxplots.pdf",
                       title: str = None) -> str:
        """
        Create box plots comparing algorithm performance.
        
        Args:
            results: Dictionary mapping algorithm name to list of metric values
            metric: Name of the metric being compared
            filename: Output filename
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        
        algorithms = list(results.keys())
        data = [results[alg] for alg in algorithms]
        colors = [self.COLORS.get(alg, '#000000') for alg in algorithms]
        
        bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style whiskers and caps
        for whisker in bp['whiskers']:
            whisker.set(color='gray', linewidth=1)
        for cap in bp['caps']:
            cap.set(color='gray', linewidth=1)
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
        
        ax.set_ylabel(self._format_objective_label(metric))
        ax.set_xlabel('Algorithm')
        
        if title:
            ax.set_title(title)
        
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        plt.xticks(rotation=45, ha='right')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def plot_adaptive_weights(self,
                               weight_history: List[Tuple[float, float, float]],
                               filename: str = "adaptive_weights.pdf",
                               title: str = "Adaptive Weight Evolution") -> str:
        """Plot evolution of adaptive decomposition weights."""
        fig, ax = plt.subplots(figsize=(6.5, 4))
        
        generations = list(range(len(weight_history)))
        slotting = [w[0] for w in weight_history]
        routing = [w[1] for w in weight_history]
        batching = [w[2] for w in weight_history]
        
        ax.stackplot(generations, slotting, routing, batching,
                    labels=['Slotting', 'Routing', 'Batching'],
                    colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                    alpha=0.8)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Weight Proportion')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(generations) - 1)
        
        if title:
            ax.set_title(title)
        
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def plot_sensitivity_heatmap(self,
                                   results: Dict[str, Dict[str, float]],
                                   filename: str = "sensitivity_heatmap.pdf",
                                   title: str = "Parameter Sensitivity Analysis") -> str:
        """
        Create heatmap for sensitivity analysis.
        
        Args:
            results: Nested dict where results[param_combo][metric] = value
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract data into matrix form
        param_combos = list(results.keys())
        if not param_combos:
            return None
            
        metrics = list(results[param_combos[0]].keys())
        
        data = np.array([
            [results[pc][m] for m in metrics]
            for pc in param_combos
        ])
        
        # Normalize columns
        data_normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
        
        im = ax.imshow(data_normalized, cmap='RdYlGn_r', aspect='auto')
        
        # Labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(param_combos)))
        ax.set_xticklabels([self._format_metric_label(m) for m in metrics])
        ax.set_yticklabels(param_combos)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Normalized Value (lower is better)', rotation=-90, va='bottom')
        
        # Add text annotations
        for i in range(len(param_combos)):
            for j in range(len(metrics)):
                ax.text(j, i, f'{data[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=7)
        
        if title:
            ax.set_title(title)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def plot_warehouse_with_routes(self,
                                    locations: List,
                                    routes: Dict[str, List[str]],
                                    filename: str = "warehouse_routes.pdf",
                                    title: str = "Warehouse Layout with Picker Routes") -> str:
        """Visualize warehouse layout with picking routes overlaid."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create location lookup
        loc_map = {loc.loc_id: loc for loc in locations}
        
        # Plot locations
        for loc in locations:
            x, y = loc.coordinates[0], loc.coordinates[1]
            
            if loc.location_type == 'depot':
                ax.scatter(x, y, c='red', s=200, marker='s', zorder=5, label='Depot')
                ax.annotate('DEPOT', (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold')
            else:
                ax.scatter(x, y, c='lightblue', s=100, marker='s', 
                          edgecolors='navy', linewidth=1, zorder=3)
        
        # Plot routes with different colors
        route_colors = plt.cm.Set1(np.linspace(0, 1, len(routes)))
        
        for (batch_id, route), color in zip(routes.items(), route_colors):
            if len(route) < 2:
                continue
            
            coords = []
            for loc_id in route:
                if loc_id in loc_map:
                    loc = loc_map[loc_id]
                    coords.append((loc.coordinates[0], loc.coordinates[1]))
            
            if len(coords) >= 2:
                xs, ys = zip(*coords)
                ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.6,
                       label=f'{batch_id}')
                
                # Add arrows
                for i in range(len(xs) - 1):
                    dx = xs[i+1] - xs[i]
                    dy = ys[i+1] - ys[i]
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        ax.annotate('', xy=(xs[i+1], ys[i+1]), 
                                   xytext=(xs[i], ys[i]),
                                   arrowprops=dict(arrowstyle='->', color=color, 
                                                  lw=1.5, alpha=0.6))
        
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle=':')
        
        if title:
            ax.set_title(title)
        
        # Legend (limit to first few routes)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 6:
            handles = handles[:6]
            labels = labels[:6]
        ax.legend(handles, labels, loc='upper right', framealpha=0.9)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def plot_hypervolume_evolution(self,
                                    histories: Dict[str, List[float]],
                                    filename: str = "hypervolume_evolution.pdf",
                                    title: str = "Hypervolume Evolution") -> str:
        """Plot hypervolume evolution over generations."""
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        
        for alg_name, hv_history in histories.items():
            generations = list(range(len(hv_history)))
            color = self.COLORS.get(alg_name, '#000000')
            ax.plot(generations, hv_history, color=color, label=alg_name, linewidth=1.5)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume')
        
        if title:
            ax.set_title(title)
        
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def plot_instance_comparison(self,
                                  results: Dict[str, Dict[str, float]],
                                  metric: str,
                                  filename: str = "instance_comparison.pdf",
                                  title: str = None) -> str:
        """
        Plot grouped bar chart comparing algorithms across instances.
        
        Args:
            results: results[algorithm][instance] = metric_value
            metric: Name of metric for axis label
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        algorithms = list(results.keys())
        instances = list(results[algorithms[0]].keys())
        
        x = np.arange(len(instances))
        width = 0.8 / len(algorithms)
        
        for i, alg in enumerate(algorithms):
            values = [results[alg][inst] for inst in instances]
            color = self.COLORS.get(alg, f'C{i}')
            ax.bar(x + i * width, values, width, label=alg, color=color, alpha=0.8)
        
        ax.set_xlabel('Instance')
        ax.set_ylabel(self._format_objective_label(metric))
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(instances, rotation=45, ha='right')
        
        if title:
            ax.set_title(title)
        
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def _format_objective_label(self, obj_name: str) -> str:
        """Format objective name for display."""
        labels = {
            'travel_distance': 'Total Travel Distance (m)',
            'throughput_time': 'Throughput Time (min)',
            'workload_balance': 'Workload Balance (CV)',
            'best_travel_distance': 'Best Travel Distance (m)',
            'best_throughput_time': 'Best Throughput Time (min)',
            'best_workload_balance': 'Best Workload Balance (CV)',
            'hypervolume': 'Hypervolume',
            'pareto_front_size': 'Pareto Front Size'
        }
        return labels.get(obj_name, obj_name.replace('_', ' ').title())
    
    def _format_metric_label(self, metric_name: str) -> str:
        """Format metric name for display."""
        return metric_name.replace('_', ' ').title()
    
    def generate_all_figures(self, experiment_results: Dict[str, Any]) -> List[str]:
        """Generate all standard figures from experiment results."""
        generated_files = []
        
        # 1. Pareto front 2D (Distance vs Time)
        if 'pareto_fronts' in experiment_results:
            generated_files.append(
                self.plot_pareto_front_2d(
                    experiment_results['pareto_fronts'],
                    ObjectiveType.TRAVEL_DISTANCE.value,
                    ObjectiveType.THROUGHPUT_TIME.value,
                    "fig1_pareto_front_dist_time.pdf",
                    "Pareto Front: Travel Distance vs Throughput Time"
                )
            )
            
            # 2. Pareto front 2D (Distance vs Balance)
            generated_files.append(
                self.plot_pareto_front_2d(
                    experiment_results['pareto_fronts'],
                    ObjectiveType.TRAVEL_DISTANCE.value,
                    ObjectiveType.WORKLOAD_BALANCE.value,
                    "fig2_pareto_front_dist_balance.pdf",
                    "Pareto Front: Travel Distance vs Workload Balance"
                )
            )
            
            # 3. 3D Pareto front
            generated_files.append(
                self.plot_pareto_front_3d(
                    experiment_results['pareto_fronts'],
                    "fig3_pareto_front_3d.pdf",
                    "3D Pareto Front"
                )
            )
        
        # 4. Convergence curves
        if 'histories' in experiment_results:
            generated_files.append(
                self.plot_convergence(
                    experiment_results['histories'],
                    'best_travel_distance',
                    "fig4_convergence.pdf",
                    "Convergence Analysis"
                )
            )
        
        # 5. Box plots
        if 'metric_results' in experiment_results:
            for metric in ['travel_distance', 'throughput_time', 'workload_balance']:
                if any(metric in r for alg_results in experiment_results['metric_results'].values() 
                       for r in alg_results):
                    results = {
                        alg: [r.get(metric, 0) for r in results_list]
                        for alg, results_list in experiment_results['metric_results'].items()
                    }
                    generated_files.append(
                        self.plot_boxplots(
                            results,
                            metric,
                            f"fig5_boxplot_{metric}.pdf",
                            f"Algorithm Comparison: {self._format_objective_label(metric)}"
                        )
                    )
        
        # 6. Adaptive weights
        if 'adaptive_weights_history' in experiment_results:
            generated_files.append(
                self.plot_adaptive_weights(
                    experiment_results['adaptive_weights_history'],
                    "fig6_adaptive_weights.pdf",
                    "Adaptive Decomposition Weight Evolution"
                )
            )
        
        # 7. Hypervolume evolution
        if 'hypervolume_histories' in experiment_results:
            generated_files.append(
                self.plot_hypervolume_evolution(
                    experiment_results['hypervolume_histories'],
                    "fig7_hypervolume.pdf",
                    "Hypervolume Evolution"
                )
            )
        
        return [f for f in generated_files if f is not None]
