"""
Comprehensive Experiment Runner for C&OR Paper v2.

Improvements:
1. Larger benchmark instances (up to 1000+ SKUs, 10000+ orders)
2. Multiple baseline algorithms (NSGA-II, MOEA/D, Sequential, ABC, Random)
3. Ablation studies (ADS, CCL components)
4. Proper hypervolume computation
5. 20-30 replications with confidence intervals
6. Comprehensive statistical analysis
"""

import os
import sys
import json
import time
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from wms_optlab.experiments.moiwof_v2 import MOIWOFv2, MOIWOFConfig, run_moiwof_v2_experiment
from wms_optlab.experiments.moiwof import ObjectiveType, ParetoSolution
from wms_optlab.experiments.baselines import (
    NSGA2Vanilla, MOEAD, SequentialOptimization, ABCHeuristic, RandomBaseline
)
from wms_optlab.experiments.benchmark_generator import (
    BenchmarkInstanceGenerator, BenchmarkInstance, InstanceConfig,
    InstanceSize, LayoutType, DemandProfile
)
from wms_optlab.experiments.hypervolume import (
    calculate_hypervolume_3d, calculate_igd, calculate_spread, QualityIndicators
)
from wms_optlab.data.models import Warehouse


@dataclass
class ExperimentResult:
    """Results from a single algorithm run."""
    algorithm: str
    instance: str
    run: int
    travel_distance: float
    throughput_time: float
    workload_balance: float
    hypervolume: float
    pareto_size: int
    spread: float
    runtime: float
    generation_history: Optional[List[Dict]] = None


@dataclass
class AblationConfig:
    """Configuration for ablation variants."""
    name: str
    enable_ads: bool
    enable_ccl: bool


class ComprehensiveExperimentRunner:
    """Runs comprehensive computational experiments."""
    
    # Extended size presets including large instances
    SIZE_PRESETS = {
        'S': {'skus': 100, 'locations': 75, 'orders': 500},
        'M': {'skus': 300, 'locations': 200, 'orders': 2000},
        'L': {'skus': 1000, 'locations': 600, 'orders': 10000},
        'XL': {'skus': 2000, 'locations': 1200, 'orders': 20000}
    }
    
    def __init__(self, output_dir: str = "experiments_output_v2"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        
        self.results: List[ExperimentResult] = []
        self.pareto_fronts: Dict[str, Dict[str, List[ParetoSolution]]] = {}
        self.reference_fronts: Dict[str, List[ParetoSolution]] = {}
    
    def generate_benchmark_suite(self, include_large: bool = True) -> List[BenchmarkInstance]:
        """Generate comprehensive benchmark suite."""
        instances = []
        
        # Small instances (detailed analysis)
        small_configs = [
            ('S-PAR-PAR', InstanceSize.SMALL, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
            ('S-PAR-UNI', InstanceSize.SMALL, LayoutType.PARALLEL_AISLE, DemandProfile.UNIFORM),
            ('S-FIS-PAR', InstanceSize.SMALL, LayoutType.FISHBONE, DemandProfile.PARETO),
            ('S-FIS-UNI', InstanceSize.SMALL, LayoutType.FISHBONE, DemandProfile.UNIFORM),
        ]
        
        # Medium instances (main comparison)
        medium_configs = [
            ('M-PAR-PAR', InstanceSize.MEDIUM, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
            ('M-PAR-UNI', InstanceSize.MEDIUM, LayoutType.PARALLEL_AISLE, DemandProfile.UNIFORM),
            ('M-FIS-PAR', InstanceSize.MEDIUM, LayoutType.FISHBONE, DemandProfile.PARETO),
        ]
        
        # Large instances (scalability)
        large_configs = [
            ('L-PAR-PAR', InstanceSize.LARGE, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
            ('L-PAR-UNI', InstanceSize.LARGE, LayoutType.PARALLEL_AISLE, DemandProfile.UNIFORM),
        ] if include_large else []
        
        for inst_id, size, layout, profile in small_configs + medium_configs + large_configs:
            config = InstanceConfig(
                size=size,
                layout_type=layout,
                demand_profile=profile,
                random_seed=42
            )
            generator = BenchmarkInstanceGenerator(config)
            instances.append(generator.generate(inst_id))
        
        return instances
    
    def run_algorithm(self, algorithm_name: str, warehouse: Warehouse, 
                     config: MOIWOFConfig, run_idx: int) -> Tuple[List[ParetoSolution], List[Dict], float]:
        """Run a single algorithm and return results."""
        config.random_seed = 42 + run_idx
        
        start_time = time.time()
        
        if algorithm_name == 'MOIWOF':
            optimizer = MOIWOFv2(warehouse, config)
            pareto_front, history = optimizer.run()
        elif algorithm_name == 'MOIWOF-NoADS':
            config_variant = MOIWOFConfig(
                population_size=config.population_size,
                max_generations=config.max_generations,
                crossover_rate=config.crossover_rate,
                mutation_rate=config.mutation_rate,
                enable_ads=False,
                enable_ccl=True,
                random_seed=config.random_seed
            )
            optimizer = MOIWOFv2(warehouse, config_variant)
            pareto_front, history = optimizer.run()
        elif algorithm_name == 'MOIWOF-NoCCL':
            config_variant = MOIWOFConfig(
                population_size=config.population_size,
                max_generations=config.max_generations,
                crossover_rate=config.crossover_rate,
                mutation_rate=config.mutation_rate,
                enable_ads=True,
                enable_ccl=False,
                random_seed=config.random_seed
            )
            optimizer = MOIWOFv2(warehouse, config_variant)
            pareto_front, history = optimizer.run()
        elif algorithm_name == 'MOIWOF-NoAdapt':
            config_variant = MOIWOFConfig(
                population_size=config.population_size,
                max_generations=config.max_generations,
                crossover_rate=config.crossover_rate,
                mutation_rate=config.mutation_rate,
                enable_ads=False,
                enable_ccl=False,
                random_seed=config.random_seed
            )
            optimizer = MOIWOFv2(warehouse, config_variant)
            pareto_front, history = optimizer.run()
        elif algorithm_name == 'NSGA-II':
            baseline = NSGA2Vanilla(warehouse, config)
            pareto_front, history = baseline.run()
        elif algorithm_name == 'MOEA/D':
            baseline = MOEAD(warehouse, config)
            pareto_front, history = baseline.run()
        elif algorithm_name == 'Sequential':
            baseline = SequentialOptimization(warehouse, config)
            pareto_front, history = baseline.run()
        elif algorithm_name == 'ABC':
            baseline = ABCHeuristic(warehouse, config)
            pareto_front, history = baseline.run()
        elif algorithm_name == 'Random':
            baseline = RandomBaseline(warehouse, config)
            pareto_front, history = baseline.run()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        runtime = time.time() - start_time
        
        return pareto_front, history, runtime
    
    def run_comprehensive_experiments(self,
                                       num_runs: int = 20,
                                       max_generations: int = 100,
                                       population_size: int = 100,
                                       include_large: bool = False,
                                       algorithms: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive experiment suite."""
        print("=" * 70)
        print("MOIWOF Comprehensive Computational Experiments v2")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Replications: {num_runs}, Generations: {max_generations}")
        print("=" * 70)
        
        if algorithms is None:
            algorithms = [
                'MOIWOF', 'MOIWOF-NoADS', 'MOIWOF-NoCCL', 'MOIWOF-NoAdapt',
                'NSGA-II', 'MOEA/D', 'Sequential', 'ABC', 'Random'
            ]
        
        # Generate benchmark instances
        print("\n1. Generating benchmark instances...")
        instances = self.generate_benchmark_suite(include_large=include_large)
        print(f"   Generated {len(instances)} instances")
        for inst in instances:
            print(f"   - {inst.instance_id}: {len(inst.warehouse.skus)} SKUs, "
                  f"{len(inst.warehouse.locations)} locations, "
                  f"{len(inst.warehouse.orders)} orders")
        
        # Configure base settings
        base_config = MOIWOFConfig(
            population_size=population_size,
            max_generations=max_generations
        )
        
        # Run experiments
        print(f"\n2. Running experiments ({len(algorithms)} algorithms × {len(instances)} instances × {num_runs} runs)")
        
        total_runs = len(algorithms) * len(instances) * num_runs
        completed = 0
        
        for inst in instances:
            inst_id = inst.instance_id
            self.pareto_fronts[inst_id] = {}
            
            # Collect all fronts for reference front computation
            all_fronts_for_reference = []
            
            for alg in algorithms:
                self.pareto_fronts[inst_id][alg] = []
                
                for run_idx in range(num_runs):
                    completed += 1
                    progress = completed / total_runs * 100
                    
                    print(f"\r   [{progress:5.1f}%] {alg} on {inst_id} run {run_idx+1}/{num_runs}...", 
                          end="", flush=True)
                    
                    try:
                        pareto_front, history, runtime = self.run_algorithm(
                            alg, inst.warehouse, base_config, run_idx
                        )
                        
                        # Calculate metrics
                        hv = calculate_hypervolume_3d(pareto_front) if pareto_front else 0.0
                        spread = calculate_spread(pareto_front) if pareto_front else 0.0
                        
                        best_dist = min((s.objectives[ObjectiveType.TRAVEL_DISTANCE.value] 
                                        for s in pareto_front), default=float('inf'))
                        best_time = min((s.objectives[ObjectiveType.THROUGHPUT_TIME.value] 
                                        for s in pareto_front), default=float('inf'))
                        best_balance = min((s.objectives[ObjectiveType.WORKLOAD_BALANCE.value] 
                                           for s in pareto_front), default=float('inf'))
                        
                        result = ExperimentResult(
                            algorithm=alg,
                            instance=inst_id,
                            run=run_idx,
                            travel_distance=best_dist,
                            throughput_time=best_time,
                            workload_balance=best_balance,
                            hypervolume=hv,
                            pareto_size=len(pareto_front),
                            spread=spread,
                            runtime=runtime,
                            generation_history=history if run_idx == 0 else None
                        )
                        self.results.append(result)
                        
                        # Store Pareto front for first run
                        if run_idx == 0:
                            self.pareto_fronts[inst_id][alg] = pareto_front
                            all_fronts_for_reference.extend(pareto_front)
                        
                    except Exception as e:
                        print(f"\n   ERROR: {alg} on {inst_id} run {run_idx}: {e}")
            
            # Compute reference front for this instance
            self.reference_fronts[inst_id] = self._compute_reference_front(all_fronts_for_reference)
            print(f"\n   {inst_id} complete. Reference front: {len(self.reference_fronts[inst_id])} solutions")
        
        print("\n\n3. Computing quality indicators with reference fronts...")
        self._compute_igd_metrics()
        
        print("\n4. Statistical analysis...")
        stats_results = self._run_statistical_analysis()
        
        print("\n5. Generating figures...")
        self._generate_figures()
        
        print("\n6. Saving results...")
        self._save_results()
        
        print("\n" + "=" * 70)
        print("Experiments completed successfully!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)
        
        return {
            'results': self.results,
            'pareto_fronts': self.pareto_fronts,
            'reference_fronts': self.reference_fronts,
            'statistics': stats_results
        }
    
    def _compute_reference_front(self, all_solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Compute reference Pareto front from all solutions."""
        if not all_solutions:
            return []
        
        non_dominated = []
        for sol in all_solutions:
            dominated = False
            for other in all_solutions:
                if other.dominates(sol):
                    dominated = True
                    break
            if not dominated:
                # Check if already in non_dominated
                is_duplicate = False
                for existing in non_dominated:
                    if (abs(sol.objectives[ObjectiveType.TRAVEL_DISTANCE.value] - 
                           existing.objectives[ObjectiveType.TRAVEL_DISTANCE.value]) < 1e-6 and
                        abs(sol.objectives[ObjectiveType.THROUGHPUT_TIME.value] - 
                           existing.objectives[ObjectiveType.THROUGHPUT_TIME.value]) < 1e-6):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    non_dominated.append(sol)
        
        return non_dominated
    
    def _compute_igd_metrics(self):
        """Compute IGD metrics using reference fronts."""
        for result in self.results:
            inst_id = result.instance
            alg = result.algorithm
            
            if inst_id in self.reference_fronts and inst_id in self.pareto_fronts:
                if alg in self.pareto_fronts[inst_id]:
                    pf = self.pareto_fronts[inst_id][alg]
                    ref = self.reference_fronts[inst_id]
                    
                    if pf and ref:
                        igd = calculate_igd(pf, ref)
                        # Store IGD (would need to add field to ExperimentResult)
    
    def _run_statistical_analysis(self) -> Dict[str, Any]:
        """Run comprehensive statistical analysis."""
        stats_results = {}
        
        # Group results by instance and algorithm
        from collections import defaultdict
        grouped = defaultdict(lambda: defaultdict(list))
        
        for r in self.results:
            grouped[r.instance][r.algorithm].append(r)
        
        metrics = ['travel_distance', 'throughput_time', 'workload_balance', 'hypervolume']
        
        for inst_id in grouped:
            stats_results[inst_id] = {}
            
            for metric in metrics:
                stats_results[inst_id][metric] = {}
                
                # Collect data for all algorithms
                alg_data = {}
                for alg, results in grouped[inst_id].items():
                    values = [getattr(r, metric) for r in results]
                    alg_data[alg] = values
                    
                    stats_results[inst_id][metric][alg] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'ci_95': stats.t.interval(0.95, len(values)-1, 
                                                   loc=np.mean(values),
                                                   scale=stats.sem(values)) if len(values) > 1 else (np.mean(values), np.mean(values))
                    }
                
                # Pairwise comparisons (MOIWOF vs others)
                if 'MOIWOF' in alg_data:
                    moiwof_data = alg_data['MOIWOF']
                    
                    for alg, data in alg_data.items():
                        if alg != 'MOIWOF' and len(data) > 1 and len(moiwof_data) > 1:
                            # Wilcoxon signed-rank test
                            try:
                                stat, p_value = stats.wilcoxon(moiwof_data[:len(data)], data[:len(moiwof_data)])
                                
                                # Effect size (Cohen's d)
                                pooled_std = np.sqrt((np.var(moiwof_data) + np.var(data)) / 2)
                                cohens_d = (np.mean(moiwof_data) - np.mean(data)) / (pooled_std + 1e-10)
                                
                                stats_results[inst_id][metric][f'MOIWOF_vs_{alg}'] = {
                                    'p_value': p_value,
                                    'significant': p_value < 0.05,
                                    'cohens_d': cohens_d,
                                    'effect_size': 'large' if abs(cohens_d) > 0.8 else 
                                                  ('medium' if abs(cohens_d) > 0.5 else 'small')
                                }
                            except:
                                pass
        
        return stats_results
    
    def _generate_figures(self):
        """Generate publication-quality figures."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            fig_dir = os.path.join(self.output_dir, "figures")
            
            # Figure 1: Pareto fronts comparison (2D: Distance vs Time)
            self._plot_pareto_2d(
                ObjectiveType.TRAVEL_DISTANCE.value,
                ObjectiveType.THROUGHPUT_TIME.value,
                os.path.join(fig_dir, "fig1_pareto_dist_time.pdf"),
                "Travel Distance vs Throughput Time"
            )
            
            # Figure 2: Pareto fronts (2D: Distance vs Balance)
            self._plot_pareto_2d(
                ObjectiveType.TRAVEL_DISTANCE.value,
                ObjectiveType.WORKLOAD_BALANCE.value,
                os.path.join(fig_dir, "fig2_pareto_dist_balance.pdf"),
                "Travel Distance vs Workload Balance"
            )
            
            # Figure 3: 3D Pareto front
            self._plot_pareto_3d(os.path.join(fig_dir, "fig3_pareto_3d.pdf"))
            
            # Figure 4: Convergence analysis
            self._plot_convergence(os.path.join(fig_dir, "fig4_convergence.pdf"))
            
            # Figure 5: Box plots comparison
            self._plot_boxplots(os.path.join(fig_dir, "fig5_boxplot_comparison.pdf"))
            
            # Figure 6: Hypervolume comparison
            self._plot_hypervolume_comparison(os.path.join(fig_dir, "fig6_hypervolume.pdf"))
            
            # Figure 7: Ablation study
            self._plot_ablation_study(os.path.join(fig_dir, "fig7_ablation.pdf"))
            
            # Figure 8: Scalability analysis
            self._plot_scalability(os.path.join(fig_dir, "fig8_scalability.pdf"))
            
            print(f"   Generated 8 figures in {fig_dir}")
            
        except ImportError as e:
            print(f"   Warning: Could not generate figures: {e}")
    
    def _plot_pareto_2d(self, obj1: str, obj2: str, filename: str, title: str):
        """Plot 2D Pareto front comparison."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select key instances
        key_instances = [inst for inst in list(self.pareto_fronts.keys())[:6]]
        
        colors = {'MOIWOF': 'red', 'NSGA-II': 'blue', 'MOEA/D': 'green', 
                 'Sequential': 'orange', 'ABC': 'purple', 'Random': 'gray'}
        markers = {'MOIWOF': 'o', 'NSGA-II': 's', 'MOEA/D': '^', 
                  'Sequential': 'D', 'ABC': 'x', 'Random': '+'}
        
        for ax_idx, inst_id in enumerate(key_instances):
            ax = axes[ax_idx]
            
            for alg in ['MOIWOF', 'NSGA-II', 'MOEA/D', 'Sequential', 'ABC', 'Random']:
                if alg in self.pareto_fronts.get(inst_id, {}):
                    pf = self.pareto_fronts[inst_id][alg]
                    if pf:
                        x = [s.objectives[obj1] for s in pf]
                        y = [s.objectives[obj2] for s in pf]
                        ax.scatter(x, y, c=colors.get(alg, 'black'), 
                                  marker=markers.get(alg, 'o'),
                                  label=alg, alpha=0.7, s=30)
            
            ax.set_xlabel(obj1.replace('_', ' ').title())
            ax.set_ylabel(obj2.replace('_', ' ').title())
            ax.set_title(inst_id)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(key_instances), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pareto_3d(self, filename: str):
        """Plot 3D Pareto front."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use first medium instance
        inst_id = None
        for inst in self.pareto_fronts:
            if inst.startswith('M-'):
                inst_id = inst
                break
        
        if inst_id is None:
            inst_id = list(self.pareto_fronts.keys())[0] if self.pareto_fronts else None
        
        if inst_id:
            colors = {'MOIWOF': 'red', 'NSGA-II': 'blue', 'MOEA/D': 'green'}
            
            for alg in ['MOIWOF', 'NSGA-II', 'MOEA/D']:
                if alg in self.pareto_fronts.get(inst_id, {}):
                    pf = self.pareto_fronts[inst_id][alg]
                    if pf:
                        x = [s.objectives[ObjectiveType.TRAVEL_DISTANCE.value] for s in pf]
                        y = [s.objectives[ObjectiveType.THROUGHPUT_TIME.value] for s in pf]
                        z = [s.objectives[ObjectiveType.WORKLOAD_BALANCE.value] for s in pf]
                        ax.scatter(x, y, z, c=colors.get(alg, 'black'), 
                                  label=alg, alpha=0.7, s=50)
            
            ax.set_xlabel('Travel Distance')
            ax.set_ylabel('Throughput Time')
            ax.set_zlabel('Workload Balance')
            ax.legend()
        
        plt.title(f'3D Pareto Front Comparison ({inst_id})')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence(self, filename: str):
        """Plot convergence analysis."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Find results with history
        histories = {}
        for r in self.results:
            if r.generation_history and r.run == 0:
                if r.algorithm not in histories:
                    histories[r.algorithm] = []
                histories[r.algorithm].append(r.generation_history)
        
        colors = {'MOIWOF': 'red', 'NSGA-II': 'blue', 'MOEA/D': 'green'}
        
        for alg, hist_list in histories.items():
            if alg in colors and hist_list:
                # Average across instances
                all_gens = []
                for hist in hist_list:
                    gens = [h.get('best_travel_distance', float('inf')) for h in hist]
                    all_gens.append(gens)
                
                # Pad to same length
                max_len = max(len(g) for g in all_gens)
                padded = []
                for g in all_gens:
                    padded.append(g + [g[-1]] * (max_len - len(g)))
                
                mean_curve = np.mean(padded, axis=0)
                ax.plot(range(len(mean_curve)), mean_curve, 
                       c=colors[alg], label=alg, linewidth=2)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Travel Distance')
        ax.set_title('Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplots(self, filename: str):
        """Plot box plots comparing algorithms."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['travel_distance', 'workload_balance', 'hypervolume']
        titles = ['Travel Distance', 'Workload Balance', 'Hypervolume']
        
        # Group data
        from collections import defaultdict
        alg_data = defaultdict(lambda: defaultdict(list))
        
        for r in self.results:
            for metric in metrics:
                alg_data[metric][r.algorithm].append(getattr(r, metric))
        
        for ax, metric, title in zip(axes, metrics, titles):
            data = []
            labels = []
            
            for alg in ['MOIWOF', 'NSGA-II', 'MOEA/D', 'Sequential', 'ABC', 'Random']:
                if alg in alg_data[metric]:
                    data.append(alg_data[metric][alg])
                    labels.append(alg)
            
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)
            
            ax.set_title(title)
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hypervolume_comparison(self, filename: str):
        """Plot hypervolume comparison across instances."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by instance
        from collections import defaultdict
        inst_alg_hv = defaultdict(lambda: defaultdict(list))
        
        for r in self.results:
            inst_alg_hv[r.instance][r.algorithm].append(r.hypervolume)
        
        instances = sorted(inst_alg_hv.keys())
        algorithms = ['MOIWOF', 'NSGA-II', 'MOEA/D', 'Sequential']
        
        x = np.arange(len(instances))
        width = 0.2
        
        colors = {'MOIWOF': 'red', 'NSGA-II': 'blue', 'MOEA/D': 'green', 'Sequential': 'orange'}
        
        for i, alg in enumerate(algorithms):
            means = []
            stds = []
            for inst in instances:
                if alg in inst_alg_hv[inst]:
                    means.append(np.mean(inst_alg_hv[inst][alg]))
                    stds.append(np.std(inst_alg_hv[inst][alg]))
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x + i * width, means, width, yerr=stds, 
                  label=alg, color=colors.get(alg, 'gray'), alpha=0.7)
        
        ax.set_xlabel('Instance')
        ax.set_ylabel('Hypervolume')
        ax.set_title('Hypervolume Comparison Across Instances')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(instances, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_study(self, filename: str):
        """Plot ablation study results."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['travel_distance', 'workload_balance', 'hypervolume']
        titles = ['Travel Distance ↓', 'Workload Balance ↓', 'Hypervolume ↑']
        
        ablation_algs = ['MOIWOF', 'MOIWOF-NoADS', 'MOIWOF-NoCCL', 'MOIWOF-NoAdapt']
        
        from collections import defaultdict
        alg_data = defaultdict(lambda: defaultdict(list))
        
        for r in self.results:
            if r.algorithm in ablation_algs:
                for metric in metrics:
                    alg_data[metric][r.algorithm].append(getattr(r, metric))
        
        for ax, metric, title in zip(axes, metrics, titles):
            data = []
            labels = []
            
            for alg in ablation_algs:
                if alg in alg_data[metric]:
                    data.append(alg_data[metric][alg])
                    labels.append(alg.replace('MOIWOF-', '').replace('MOIWOF', 'Full'))
            
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                
                colors = ['red', 'orange', 'yellow', 'gray']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
            
            ax.set_title(title)
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Ablation Study: Component Contributions', fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability(self, filename: str):
        """Plot scalability analysis."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Group by instance size
        from collections import defaultdict
        size_runtime = defaultdict(lambda: defaultdict(list))
        size_hv = defaultdict(lambda: defaultdict(list))
        
        for r in self.results:
            size = r.instance.split('-')[0]  # S, M, L
            size_runtime[size][r.algorithm].append(r.runtime)
            size_hv[size][r.algorithm].append(r.hypervolume)
        
        # Runtime plot
        ax = axes[0]
        sizes = ['S', 'M', 'L']
        algorithms = ['MOIWOF', 'NSGA-II', 'MOEA/D']
        
        for alg in algorithms:
            means = []
            for size in sizes:
                if size in size_runtime and alg in size_runtime[size]:
                    means.append(np.mean(size_runtime[size][alg]))
                else:
                    means.append(0)
            ax.plot(sizes[:len(means)], means, 'o-', label=alg, linewidth=2, markersize=8)
        
        ax.set_xlabel('Instance Size')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Scalability: Runtime')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Hypervolume plot
        ax = axes[1]
        for alg in algorithms:
            means = []
            for size in sizes:
                if size in size_hv and alg in size_hv[size]:
                    means.append(np.mean(size_hv[size][alg]))
                else:
                    means.append(0)
            ax.plot(sizes[:len(means)], means, 'o-', label=alg, linewidth=2, markersize=8)
        
        ax.set_xlabel('Instance Size')
        ax.set_ylabel('Hypervolume')
        ax.set_title('Scalability: Solution Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save all results to files."""
        # Save detailed CSV
        csv_path = os.path.join(self.output_dir, 'data', 'detailed_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Algorithm', 'Instance', 'Run', 'Travel_Distance', 
                           'Throughput_Time', 'Workload_Balance', 'Hypervolume',
                           'Pareto_Size', 'Spread', 'Runtime'])
            for r in self.results:
                writer.writerow([
                    r.algorithm, r.instance, r.run, r.travel_distance,
                    r.throughput_time, r.workload_balance, r.hypervolume,
                    r.pareto_size, r.spread, r.runtime
                ])
        
        # Save summary JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_runs': len(self.results),
            'instances': list(self.pareto_fronts.keys()),
            'algorithms': list(set(r.algorithm for r in self.results)),
            'aggregate_results': {}
        }
        
        from collections import defaultdict
        alg_metrics = defaultdict(lambda: defaultdict(list))
        
        for r in self.results:
            alg_metrics[r.algorithm]['travel_distance'].append(r.travel_distance)
            alg_metrics[r.algorithm]['workload_balance'].append(r.workload_balance)
            alg_metrics[r.algorithm]['hypervolume'].append(r.hypervolume)
            alg_metrics[r.algorithm]['runtime'].append(r.runtime)
        
        for alg, metrics in alg_metrics.items():
            summary['aggregate_results'][alg] = {
                'mean_travel_distance': float(np.mean(metrics['travel_distance'])),
                'std_travel_distance': float(np.std(metrics['travel_distance'])),
                'mean_workload_balance': float(np.mean(metrics['workload_balance'])),
                'std_workload_balance': float(np.std(metrics['workload_balance'])),
                'mean_hypervolume': float(np.mean(metrics['hypervolume'])),
                'std_hypervolume': float(np.std(metrics['hypervolume'])),
                'mean_runtime': float(np.mean(metrics['runtime']))
            }
        
        with open(os.path.join(self.output_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   Saved results to {self.output_dir}")


def main():
    """Main entry point for comprehensive experiments."""
    runner = ComprehensiveExperimentRunner("experiments_output_v2")
    
    # Run with reasonable parameters for testing
    # For paper: num_runs=20, max_generations=150, include_large=True
    results = runner.run_comprehensive_experiments(
        num_runs=5,              # Reduce for testing
        max_generations=50,       # Reduce for testing  
        population_size=50,       # Reduce for testing
        include_large=False,      # Skip large for testing
        algorithms=['MOIWOF', 'MOIWOF-NoADS', 'MOIWOF-NoCCL', 
                   'NSGA-II', 'MOEA/D', 'Sequential', 'ABC', 'Random']
    )
    
    return results


if __name__ == "__main__":
    main()
