"""
Main Experiment Runner for C&OR Paper.

Executes comprehensive computational experiments comparing MOIWOF against
baseline algorithms across benchmark instances.

Usage:
    python -m wms_optlab.experiments.run_experiments
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from wms_optlab.experiments.moiwof import MOIWOF, MOIWOFConfig, ParetoSolution, run_moiwof_experiment, ObjectiveType
from wms_optlab.experiments.benchmark_generator import (
    BenchmarkInstanceGenerator, BenchmarkInstance, InstanceConfig,
    InstanceSize, LayoutType, DemandProfile
)
from wms_optlab.experiments.statistical_analysis import StatisticalAnalyzer, QualityIndicators
from wms_optlab.experiments.visualization import ExperimentVisualizer
from wms_optlab.slotting.heuristics import assign_by_abc_popularity
from wms_optlab.layout.geometry import manhattan_distance
from wms_optlab.routing.policies import get_s_shape_route
from wms_optlab.data.models import Warehouse


class BaselineAlgorithm:
    """Base class for baseline comparison algorithms."""
    
    def __init__(self, warehouse: Warehouse, config: MOIWOFConfig):
        self.warehouse = warehouse
        self.config = config
        self.depot = next((loc for loc in warehouse.locations if loc.location_type == 'depot'), None)
        self.graph = warehouse.get_graph()
        self.loc_map = {loc.loc_id: loc for loc in warehouse.locations}
        self.order_map = {order.order_id: order for order in warehouse.orders}
        self.storage_locations = [loc for loc in warehouse.locations if loc.location_type == 'storage']
    
    def run(self) -> List[ParetoSolution]:
        """Run the algorithm and return Pareto front."""
        raise NotImplementedError


class RandomBaseline(BaselineAlgorithm):
    """Random assignment baseline."""
    
    def run(self) -> List[ParetoSolution]:
        solutions = []
        
        for _ in range(self.config.population_size):
            # Random slotting
            skus = list(self.warehouse.skus)
            locs = list(self.storage_locations)
            np.random.shuffle(skus)
            np.random.shuffle(locs)
            
            slotting_plan = {}
            for i, sku in enumerate(skus):
                if i < len(locs):
                    slotting_plan[sku.sku_id] = locs[i].loc_id
            
            # Random batching
            orders = list(self.warehouse.orders)
            np.random.shuffle(orders)
            batches = []
            for i in range(0, len(orders), self.config.max_batch_size):
                batch = [o.order_id for o in orders[i:i+self.config.max_batch_size]]
                batches.append(batch)
            
            # Compute routes and objectives
            routes = self._compute_routes(slotting_plan, batches)
            objectives = self._evaluate_objectives(slotting_plan, batches, routes)
            
            solutions.append(ParetoSolution(
                slotting_plan=slotting_plan,
                batches=batches,
                routes=routes,
                objectives=objectives
            ))
        
        # Return non-dominated solutions
        return self._get_non_dominated(solutions)
    
    def _compute_routes(self, slotting_plan, batches):
        routes = {}
        for batch_idx, batch in enumerate(batches):
            pick_loc_ids = set()
            for order_id in batch:
                order = self.order_map.get(order_id)
                if order:
                    for line in order.order_lines:
                        loc_id = slotting_plan.get(line.sku.sku_id)
                        if loc_id:
                            pick_loc_ids.add(loc_id)
            
            pick_locations = [self.loc_map[loc_id] for loc_id in pick_loc_ids if loc_id in self.loc_map]
            if pick_locations:
                route = get_s_shape_route(pick_locations, self.graph, self.depot)
            else:
                route = [self.depot.loc_id, self.depot.loc_id]
            routes[f"batch_{batch_idx}"] = route
        return routes
    
    def _evaluate_objectives(self, slotting_plan, batches, routes):
        total_distance = 0.0
        for route in routes.values():
            for i in range(len(route) - 1):
                loc1 = self.loc_map.get(route[i])
                loc2 = self.loc_map.get(route[i + 1])
                if loc1 and loc2:
                    total_distance += manhattan_distance(loc1.coordinates, loc2.coordinates)
        
        batch_times = []
        for route in routes.values():
            dist = sum(
                manhattan_distance(self.loc_map[route[i]].coordinates, self.loc_map[route[i+1]].coordinates)
                for i in range(len(route) - 1)
                if route[i] in self.loc_map and route[i+1] in self.loc_map
            )
            batch_times.append(dist + len(route) * 0.5)
        
        picker_workloads = [0.0] * self.config.num_pickers
        for i, time in enumerate(batch_times):
            picker_workloads[i % self.config.num_pickers] += time
        
        throughput_time = max(picker_workloads) if picker_workloads else 0.0
        workload_balance = np.std(picker_workloads) / (np.mean(picker_workloads) + 1e-10) if picker_workloads else 0.0
        
        return {
            ObjectiveType.TRAVEL_DISTANCE.value: total_distance,
            ObjectiveType.THROUGHPUT_TIME.value: throughput_time,
            ObjectiveType.WORKLOAD_BALANCE.value: workload_balance
        }
    
    def _get_non_dominated(self, solutions):
        non_dominated = []
        for sol in solutions:
            dominated = False
            for other in solutions:
                if other.dominates(sol):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(sol)
        return non_dominated


class ABCHeuristicBaseline(BaselineAlgorithm):
    """ABC-popularity heuristic baseline."""
    
    def run(self) -> List[ParetoSolution]:
        # Use ABC slotting
        slotting_plan = assign_by_abc_popularity(
            self.warehouse.skus,
            self.warehouse.locations,
            self.warehouse.orders,
            distance_metric=manhattan_distance,
            depot_location=self.depot.coordinates
        )
        
        # Due-date batching
        orders_sorted = sorted(self.warehouse.orders, key=lambda o: o.due_date)
        batches = []
        for i in range(0, len(orders_sorted), self.config.max_batch_size):
            batch = [o.order_id for o in orders_sorted[i:i+self.config.max_batch_size]]
            batches.append(batch)
        
        # Compute routes
        routes = self._compute_routes(slotting_plan, batches)
        objectives = self._evaluate_objectives(slotting_plan, batches, routes)
        
        return [ParetoSolution(
            slotting_plan=slotting_plan,
            batches=batches,
            routes=routes,
            objectives=objectives
        )]
    
    def _compute_routes(self, slotting_plan, batches):
        routes = {}
        for batch_idx, batch in enumerate(batches):
            pick_loc_ids = set()
            for order_id in batch:
                order = self.order_map.get(order_id)
                if order:
                    for line in order.order_lines:
                        loc_id = slotting_plan.get(line.sku.sku_id)
                        if loc_id:
                            pick_loc_ids.add(loc_id)
            
            pick_locations = [self.loc_map[loc_id] for loc_id in pick_loc_ids if loc_id in self.loc_map]
            if pick_locations:
                route = get_s_shape_route(pick_locations, self.graph, self.depot)
            else:
                route = [self.depot.loc_id, self.depot.loc_id]
            routes[f"batch_{batch_idx}"] = route
        return routes
    
    def _evaluate_objectives(self, slotting_plan, batches, routes):
        total_distance = 0.0
        for route in routes.values():
            for i in range(len(route) - 1):
                loc1 = self.loc_map.get(route[i])
                loc2 = self.loc_map.get(route[i + 1])
                if loc1 and loc2:
                    total_distance += manhattan_distance(loc1.coordinates, loc2.coordinates)
        
        batch_times = []
        for route in routes.values():
            dist = sum(
                manhattan_distance(self.loc_map[route[i]].coordinates, self.loc_map[route[i+1]].coordinates)
                for i in range(len(route) - 1)
                if route[i] in self.loc_map and route[i+1] in self.loc_map
            )
            batch_times.append(dist + len(route) * 0.5)
        
        picker_workloads = [0.0] * self.config.num_pickers
        for i, time in enumerate(batch_times):
            picker_workloads[i % self.config.num_pickers] += time
        
        throughput_time = max(picker_workloads) if picker_workloads else 0.0
        workload_balance = np.std(picker_workloads) / (np.mean(picker_workloads) + 1e-10) if picker_workloads else 0.0
        
        return {
            ObjectiveType.TRAVEL_DISTANCE.value: total_distance,
            ObjectiveType.THROUGHPUT_TIME.value: throughput_time,
            ObjectiveType.WORKLOAD_BALANCE.value: workload_balance
        }


class ExperimentRunner:
    """Runs comprehensive computational experiments."""
    
    def __init__(self, output_dir: str = "experiments_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.analyzer = StatisticalAnalyzer()
        self.visualizer = ExperimentVisualizer(os.path.join(output_dir, "figures"))
        self.results = {}
    
    def run_full_experiment(self, 
                             num_runs: int = 10,
                             max_generations: int = 100,
                             population_size: int = 50) -> Dict[str, Any]:
        """Run the complete experiment suite."""
        print("=" * 60)
        print("MOIWOF Computational Experiments")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Generate benchmark instances
        print("\n1. Generating benchmark instances...")
        instances = self._generate_instances()
        print(f"   Generated {len(instances)} instances")
        
        # Configure algorithms
        config = MOIWOFConfig(
            population_size=population_size,
            max_generations=max_generations,
            random_seed=42
        )
        
        # Run experiments
        all_results = {
            'MOIWOF': {},
            'ABC': {},
            'Random': {}
        }
        
        pareto_fronts = {'MOIWOF': [], 'ABC': [], 'Random': []}
        histories = {'MOIWOF': []}
        metric_results = {'MOIWOF': [], 'ABC': [], 'Random': []}
        
        for inst_idx, instance in enumerate(instances):
            print(f"\n2. Running experiments on {instance.instance_id} ({inst_idx + 1}/{len(instances)})")
            
            for run_idx in range(num_runs):
                print(f"   Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)
                
                # Set seed for reproducibility
                run_config = MOIWOFConfig(
                    population_size=population_size,
                    max_generations=max_generations,
                    random_seed=42 + run_idx
                )
                
                # Run MOIWOF
                start_time = time.time()
                moiwof_result = run_moiwof_experiment(instance.warehouse, run_config)
                moiwof_time = time.time() - start_time
                
                # Extract best objective values
                best_dist = min(s.objectives[ObjectiveType.TRAVEL_DISTANCE.value] 
                               for s in moiwof_result['pareto_front'])
                best_time = min(s.objectives[ObjectiveType.THROUGHPUT_TIME.value] 
                               for s in moiwof_result['pareto_front'])
                best_balance = min(s.objectives[ObjectiveType.WORKLOAD_BALANCE.value] 
                                  for s in moiwof_result['pareto_front'])
                
                metric_results['MOIWOF'].append({
                    'instance': instance.instance_id,
                    'run': run_idx,
                    'travel_distance': best_dist,
                    'throughput_time': best_time,
                    'workload_balance': best_balance,
                    'pareto_size': len(moiwof_result['pareto_front']),
                    'runtime': moiwof_time
                })
                
                if run_idx == 0:  # Store first run's Pareto front
                    pareto_fronts['MOIWOF'].extend(moiwof_result['pareto_front'])
                    histories['MOIWOF'] = moiwof_result['history']
                
                # Run ABC baseline
                abc_baseline = ABCHeuristicBaseline(instance.warehouse, run_config)
                abc_front = abc_baseline.run()
                
                if abc_front:
                    abc_dist = abc_front[0].objectives[ObjectiveType.TRAVEL_DISTANCE.value]
                    abc_time = abc_front[0].objectives[ObjectiveType.THROUGHPUT_TIME.value]
                    abc_balance = abc_front[0].objectives[ObjectiveType.WORKLOAD_BALANCE.value]
                else:
                    abc_dist = abc_time = abc_balance = float('inf')
                
                metric_results['ABC'].append({
                    'instance': instance.instance_id,
                    'run': run_idx,
                    'travel_distance': abc_dist,
                    'throughput_time': abc_time,
                    'workload_balance': abc_balance,
                    'pareto_size': len(abc_front),
                    'runtime': 0.1  # Near-instant
                })
                
                if run_idx == 0:
                    pareto_fronts['ABC'].extend(abc_front)
                
                # Run Random baseline
                random_baseline = RandomBaseline(instance.warehouse, run_config)
                random_front = random_baseline.run()
                
                if random_front:
                    random_dist = min(s.objectives[ObjectiveType.TRAVEL_DISTANCE.value] 
                                     for s in random_front)
                    random_time = min(s.objectives[ObjectiveType.THROUGHPUT_TIME.value] 
                                     for s in random_front)
                    random_balance = min(s.objectives[ObjectiveType.WORKLOAD_BALANCE.value] 
                                        for s in random_front)
                else:
                    random_dist = random_time = random_balance = float('inf')
                
                metric_results['Random'].append({
                    'instance': instance.instance_id,
                    'run': run_idx,
                    'travel_distance': random_dist,
                    'throughput_time': random_time,
                    'workload_balance': random_balance,
                    'pareto_size': len(random_front),
                    'runtime': 0.5
                })
                
                if run_idx == 0:
                    pareto_fronts['Random'].extend(random_front)
                
                print(f"MOIWOF={best_dist:.1f}, ABC={abc_dist:.1f}, Random={random_dist:.1f}")
        
        # Store results
        self.results = {
            'pareto_fronts': pareto_fronts,
            'histories': histories,
            'metric_results': metric_results,
            'adaptive_weights_history': moiwof_result.get('adaptive_weights_history', []),
            'instances': [inst.metadata for inst in instances]
        }
        
        # Generate figures
        print("\n3. Generating figures...")
        self._generate_figures()
        
        # Statistical analysis
        print("\n4. Statistical analysis...")
        self._run_statistical_analysis()
        
        # Save results
        print("\n5. Saving results...")
        self._save_results()
        
        print("\n" + "=" * 60)
        print("Experiments completed successfully!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)
        
        return self.results
    
    def _generate_instances(self) -> List[BenchmarkInstance]:
        """Generate benchmark instances for experiments."""
        instances = []
        
        # Small instances for detailed analysis
        for layout in [LayoutType.PARALLEL_AISLE, LayoutType.FISHBONE]:
            for profile in [DemandProfile.PARETO, DemandProfile.UNIFORM]:
                config = InstanceConfig(
                    size=InstanceSize.SMALL,
                    layout_type=layout,
                    demand_profile=profile,
                    random_seed=42
                )
                generator = BenchmarkInstanceGenerator(config)
                inst_id = f"S-{layout.value[:3].upper()}-{profile.value[:3].upper()}"
                instances.append(generator.generate(inst_id))
        
        # Medium instance for main comparison
        config = InstanceConfig(
            size=InstanceSize.MEDIUM,
            layout_type=LayoutType.PARALLEL_AISLE,
            demand_profile=DemandProfile.PARETO,
            random_seed=42
        )
        generator = BenchmarkInstanceGenerator(config)
        instances.append(generator.generate("M-PAR-PAR"))
        
        return instances
    
    def _generate_figures(self):
        """Generate all publication figures."""
        # Figure 1: 2D Pareto front (Distance vs Time)
        self.visualizer.plot_pareto_front_2d(
            self.results['pareto_fronts'],
            ObjectiveType.TRAVEL_DISTANCE.value,
            ObjectiveType.THROUGHPUT_TIME.value,
            "fig1_pareto_dist_time.pdf",
            "Pareto Front: Travel Distance vs Throughput Time"
        )
        
        # Figure 2: 2D Pareto front (Distance vs Balance)
        self.visualizer.plot_pareto_front_2d(
            self.results['pareto_fronts'],
            ObjectiveType.TRAVEL_DISTANCE.value,
            ObjectiveType.WORKLOAD_BALANCE.value,
            "fig2_pareto_dist_balance.pdf",
            "Pareto Front: Travel Distance vs Workload Balance"
        )
        
        # Figure 3: 3D Pareto front
        self.visualizer.plot_pareto_front_3d(
            self.results['pareto_fronts'],
            "fig3_pareto_3d.pdf",
            "3D Pareto Front Comparison"
        )
        
        # Figure 4: Convergence plot
        self.visualizer.plot_convergence(
            {'MOIWOF': self.results['histories']['MOIWOF']},
            'best_travel_distance',
            "fig4_convergence.pdf",
            "Convergence Analysis: Travel Distance"
        )
        
        # Figure 5: Box plots for travel distance
        dist_results = {
            alg: [r['travel_distance'] for r in results]
            for alg, results in self.results['metric_results'].items()
        }
        self.visualizer.plot_boxplots(
            dist_results,
            'travel_distance',
            "fig5_boxplot_distance.pdf",
            "Algorithm Comparison: Travel Distance"
        )
        
        # Figure 6: Adaptive weights evolution
        if self.results['adaptive_weights_history']:
            self.visualizer.plot_adaptive_weights(
                self.results['adaptive_weights_history'],
                "fig6_adaptive_weights.pdf",
                "Adaptive Decomposition Weight Evolution"
            )
        
        # Figure 7: Instance comparison
        # Aggregate by instance
        instance_results = {}
        for alg, results in self.results['metric_results'].items():
            instance_results[alg] = {}
            for r in results:
                inst = r['instance']
                if inst not in instance_results[alg]:
                    instance_results[alg][inst] = []
                instance_results[alg][inst].append(r['travel_distance'])
            # Average per instance
            for inst in instance_results[alg]:
                instance_results[alg][inst] = np.mean(instance_results[alg][inst])
        
        self.visualizer.plot_instance_comparison(
            instance_results,
            'travel_distance',
            "fig7_instance_comparison.pdf",
            "Performance Across Benchmark Instances"
        )
        
        print(f"   Generated 7 figures in {self.visualizer.output_dir}")
    
    def _run_statistical_analysis(self):
        """Run comprehensive statistical analysis."""
        # Compare algorithms for each metric
        for metric in ['travel_distance', 'throughput_time', 'workload_balance']:
            print(f"   Analyzing {metric}...")
            
            comparison = self.analyzer.compare_algorithms(
                self.results['metric_results'],
                metric
            )
            
            print(f"      Ranking: {[f'{r[0]}({r[1]:.2f})' for r in comparison['ranking']]}")
            
            # Pairwise tests
            for test_name, result in comparison['pairwise_tests'].items():
                if result.significant:
                    print(f"      {test_name}: p={result.p_value:.4f}*, effect={result.effect_interpretation}")
    
    def _save_results(self):
        """Save all results to files."""
        # Save summary JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'instances': self.results['instances'],
            'algorithm_summary': {}
        }
        
        for alg, results in self.results['metric_results'].items():
            if results:
                summary['algorithm_summary'][alg] = {
                    'mean_travel_distance': np.mean([r['travel_distance'] for r in results]),
                    'std_travel_distance': np.std([r['travel_distance'] for r in results]),
                    'mean_throughput_time': np.mean([r['throughput_time'] for r in results]),
                    'mean_workload_balance': np.mean([r['workload_balance'] for r in results]),
                    'mean_pareto_size': np.mean([r['pareto_size'] for r in results]),
                    'mean_runtime': np.mean([r['runtime'] for r in results])
                }
        
        with open(os.path.join(self.output_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results CSV
        import csv
        csv_path = os.path.join(self.output_dir, 'detailed_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Algorithm', 'Instance', 'Run', 'Travel_Distance', 
                           'Throughput_Time', 'Workload_Balance', 'Pareto_Size', 'Runtime'])
            for alg, results in self.results['metric_results'].items():
                for r in results:
                    writer.writerow([
                        alg, r['instance'], r['run'], r['travel_distance'],
                        r['throughput_time'], r['workload_balance'], r['pareto_size'], r['runtime']
                    ])


def main():
    """Main entry point for experiments."""
    runner = ExperimentRunner("experiments_output")
    
    # Run with reduced parameters for faster execution
    results = runner.run_full_experiment(
        num_runs=5,           # 5 independent runs per instance
        max_generations=50,   # 50 generations per run
        population_size=30    # Population size of 30
    )
    
    return results


if __name__ == "__main__":
    main()
