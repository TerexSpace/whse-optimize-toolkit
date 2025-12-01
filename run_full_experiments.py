#!/usr/bin/env python
"""
Run comprehensive experiments for C&OR paper.
"""

import json
import time
import os
import sys
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wms_optlab.experiments.moiwof_v2 import MOIWOFv2, MOIWOFConfig
from wms_optlab.experiments.baselines import (
    ABCHeuristic, NSGA2Vanilla, MOEAD, 
    SequentialOptimization, RandomBaseline
)
from wms_optlab.experiments.benchmark_generator import (
    BenchmarkInstanceGenerator, InstanceConfig, 
    InstanceSize, LayoutType, DemandProfile
)
from wms_optlab.experiments.hypervolume import calculate_hypervolume_3d


def run_experiments():
    print('=' * 70)
    print('MOIWOF COMPREHENSIVE EXPERIMENTS FOR C&OR PAPER')
    print('=' * 70)
    
    # Configuration
    NUM_RUNS = 5
    MAX_GENERATIONS = 50
    POPULATION_SIZE = 50
    
    # Reference point for hypervolume
    ref_point = {
        'travel_distance': 50000, 
        'throughput_time': 10000, 
        'workload_balance': 1.0
    }
    
    # Instances to test
    instances_config = [
        ('S-PAR-P', InstanceSize.SMALL, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
        ('S-PAR-U', InstanceSize.SMALL, LayoutType.PARALLEL_AISLE, DemandProfile.UNIFORM),
        ('S-FIS-P', InstanceSize.SMALL, LayoutType.FISHBONE, DemandProfile.PARETO),
        ('M-PAR-P', InstanceSize.MEDIUM, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
        ('M-PAR-U', InstanceSize.MEDIUM, LayoutType.PARALLEL_AISLE, DemandProfile.UNIFORM),
        ('M-FIS-P', InstanceSize.MEDIUM, LayoutType.FISHBONE, DemandProfile.PARETO),
    ]
    
    # Store all results
    all_results = defaultdict(dict)
    
    for inst_name, size, layout, demand in instances_config:
        print(f'\nInstance: {inst_name} ({size.value}, {layout.value}, {demand.value})')
        
        # Generate instance
        config = InstanceConfig(
            size=size, 
            layout_type=layout, 
            demand_profile=demand, 
            random_seed=42
        )
        gen = BenchmarkInstanceGenerator(config)
        inst = gen.generate(inst_name)
        print(f'  SKUs: {len(inst.warehouse.skus)}, Locations: {len(inst.warehouse.locations)}, Orders: {len(inst.warehouse.orders)}')
        
        # Define algorithms for this instance
        def create_algorithms(warehouse, algo_config):
            return {
                'MOIWOF': MOIWOFv2(warehouse, algo_config),
                'MOIWOF-NoADS': MOIWOFv2(warehouse, MOIWOFConfig(
                    population_size=algo_config.population_size,
                    max_generations=algo_config.max_generations,
                    enable_ads=False, enable_ccl=True,
                    random_seed=algo_config.random_seed
                )),
                'MOIWOF-NoCCL': MOIWOFv2(warehouse, MOIWOFConfig(
                    population_size=algo_config.population_size,
                    max_generations=algo_config.max_generations,
                    enable_ads=True, enable_ccl=False,
                    random_seed=algo_config.random_seed
                )),
                'NSGA-II': NSGA2Vanilla(warehouse, algo_config),
                'MOEA/D': MOEAD(warehouse, algo_config),
                'Sequential': SequentialOptimization(warehouse, algo_config),
                'ABC': ABCHeuristic(warehouse, algo_config),
                'Random': RandomBaseline(warehouse, algo_config),
            }
        
        alg_names = ['MOIWOF', 'MOIWOF-NoADS', 'MOIWOF-NoCCL', 'NSGA-II', 'MOEA/D', 'Sequential', 'ABC', 'Random']
        
        for alg_name in alg_names:
            alg_results = {
                'distance': [], 'balance': [], 'throughput': [], 
                'hypervolume': [], 'runtime': [], 'pareto_size': []
            }
            
            for run in range(NUM_RUNS):
                algo_config = MOIWOFConfig(
                    population_size=POPULATION_SIZE,
                    max_generations=MAX_GENERATIONS,
                    random_seed=42 + run
                )
                
                # Create fresh algorithm instance
                algorithms = create_algorithms(inst.warehouse, algo_config)
                alg = algorithms[alg_name]
                
                start = time.time()
                pareto_front, _ = alg.run()
                runtime = time.time() - start
                
                best_dist = min(p.objectives['travel_distance'] for p in pareto_front)
                best_balance = min(p.objectives['workload_balance'] for p in pareto_front)
                best_throughput = min(p.objectives['throughput_time'] for p in pareto_front)
                hv = calculate_hypervolume_3d(pareto_front, ref_point)
                
                alg_results['distance'].append(best_dist)
                alg_results['balance'].append(best_balance)
                alg_results['throughput'].append(best_throughput)
                alg_results['hypervolume'].append(hv)
                alg_results['runtime'].append(runtime)
                alg_results['pareto_size'].append(len(pareto_front))
            
            # Store aggregated results
            all_results[inst_name][alg_name] = {
                'distance_mean': float(np.mean(alg_results['distance'])),
                'distance_std': float(np.std(alg_results['distance'])),
                'balance_mean': float(np.mean(alg_results['balance'])),
                'balance_std': float(np.std(alg_results['balance'])),
                'throughput_mean': float(np.mean(alg_results['throughput'])),
                'throughput_std': float(np.std(alg_results['throughput'])),
                'hypervolume_mean': float(np.mean(alg_results['hypervolume'])),
                'hypervolume_std': float(np.std(alg_results['hypervolume'])),
                'runtime_mean': float(np.mean(alg_results['runtime'])),
                'pareto_size_mean': float(np.mean(alg_results['pareto_size'])),
            }
            
            dist_mean = np.mean(alg_results['distance'])
            dist_std = np.std(alg_results['distance'])
            hv_mean = np.mean(alg_results['hypervolume'])
            print(f'  {alg_name}: Dist={dist_mean:.1f}+/-{dist_std:.1f}, HV={hv_mean:.2e}')
    
    # Create output directory if needed
    os.makedirs('experiments_output_v2', exist_ok=True)
    
    # Save results to JSON
    with open('experiments_output_v2/full_results.json', 'w') as f:
        json.dump(dict(all_results), f, indent=2)
    
    print('\n' + '=' * 70)
    print('EXPERIMENTS COMPLETE')
    print('Results saved to experiments_output_v2/full_results.json')
    print('=' * 70)
    
    return all_results


if __name__ == "__main__":
    run_experiments()
