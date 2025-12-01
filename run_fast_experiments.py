#!/usr/bin/env python
"""
Run efficient experiments for C&OR paper - optimized for faster execution.
"""

import json
import time
import os
import sys
import numpy as np
from collections import defaultdict

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


def run_single_experiment(inst, alg_name, algo_config, ref_point):
    """Run a single algorithm on a single instance."""
    warehouse = inst.warehouse
    
    if alg_name == 'MOIWOF':
        alg = MOIWOFv2(warehouse, algo_config)
    elif alg_name == 'MOIWOF-NoADS':
        cfg = MOIWOFConfig(
            population_size=algo_config.population_size,
            max_generations=algo_config.max_generations,
            enable_ads=False, enable_ccl=True,
            random_seed=algo_config.random_seed
        )
        alg = MOIWOFv2(warehouse, cfg)
    elif alg_name == 'MOIWOF-NoCCL':
        cfg = MOIWOFConfig(
            population_size=algo_config.population_size,
            max_generations=algo_config.max_generations,
            enable_ads=True, enable_ccl=False,
            random_seed=algo_config.random_seed
        )
        alg = MOIWOFv2(warehouse, cfg)
    elif alg_name == 'NSGA-II':
        alg = NSGA2Vanilla(warehouse, algo_config)
    elif alg_name == 'MOEA/D':
        alg = MOEAD(warehouse, algo_config)
    elif alg_name == 'Sequential':
        alg = SequentialOptimization(warehouse, algo_config)
    elif alg_name == 'ABC':
        alg = ABCHeuristic(warehouse, algo_config)
    elif alg_name == 'Random':
        alg = RandomBaseline(warehouse, algo_config)
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")
    
    start = time.time()
    pareto_front, _ = alg.run()
    runtime = time.time() - start
    
    best_dist = min(p.objectives['travel_distance'] for p in pareto_front)
    best_balance = min(p.objectives['workload_balance'] for p in pareto_front)
    best_throughput = min(p.objectives['throughput_time'] for p in pareto_front)
    hv = calculate_hypervolume_3d(pareto_front, ref_point)
    
    return {
        'distance': best_dist,
        'balance': best_balance,
        'throughput': best_throughput,
        'hypervolume': hv,
        'runtime': runtime,
        'pareto_size': len(pareto_front)
    }


def run_experiments():
    print('=' * 70)
    print('MOIWOF EXPERIMENTS FOR C&OR PAPER')
    print('=' * 70)
    
    # Faster configuration for meaningful results
    NUM_RUNS = 3
    MAX_GENERATIONS = 30
    POPULATION_SIZE = 40
    
    ref_point = {
        'travel_distance': 100000, 
        'throughput_time': 20000, 
        'workload_balance': 1.0
    }
    
    # Test instances
    instances_config = [
        ('S-PAR-P', InstanceSize.SMALL, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
        ('S-PAR-U', InstanceSize.SMALL, LayoutType.PARALLEL_AISLE, DemandProfile.UNIFORM),
        ('S-FIS-P', InstanceSize.SMALL, LayoutType.FISHBONE, DemandProfile.PARETO),
        ('M-PAR-P', InstanceSize.MEDIUM, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
    ]
    
    # Algorithms
    alg_names = ['MOIWOF', 'MOIWOF-NoADS', 'MOIWOF-NoCCL', 'NSGA-II', 'MOEA/D', 'Sequential', 'ABC', 'Random']
    
    all_results = {}
    
    for inst_name, size, layout, demand in instances_config:
        print(f'\n{"="*50}')
        print(f'Instance: {inst_name}')
        print(f'{"="*50}')
        
        config = InstanceConfig(size=size, layout_type=layout, demand_profile=demand, random_seed=42)
        gen = BenchmarkInstanceGenerator(config)
        inst = gen.generate(inst_name)
        print(f'SKUs: {len(inst.warehouse.skus)}, Locations: {len(inst.warehouse.locations)}, Orders: {len(inst.warehouse.orders)}')
        
        all_results[inst_name] = {}
        
        for alg_name in alg_names:
            results_list = []
            
            for run in range(NUM_RUNS):
                algo_config = MOIWOFConfig(
                    population_size=POPULATION_SIZE,
                    max_generations=MAX_GENERATIONS,
                    random_seed=42 + run
                )
                
                try:
                    result = run_single_experiment(inst, alg_name, algo_config, ref_point)
                    results_list.append(result)
                except Exception as e:
                    print(f'  {alg_name} run {run+1} failed: {e}')
                    continue
            
            if results_list:
                all_results[inst_name][alg_name] = {
                    'distance_mean': float(np.mean([r['distance'] for r in results_list])),
                    'distance_std': float(np.std([r['distance'] for r in results_list])),
                    'balance_mean': float(np.mean([r['balance'] for r in results_list])),
                    'balance_std': float(np.std([r['balance'] for r in results_list])),
                    'throughput_mean': float(np.mean([r['throughput'] for r in results_list])),
                    'hypervolume_mean': float(np.mean([r['hypervolume'] for r in results_list])),
                    'hypervolume_std': float(np.std([r['hypervolume'] for r in results_list])),
                    'runtime_mean': float(np.mean([r['runtime'] for r in results_list])),
                    'pareto_size_mean': float(np.mean([r['pareto_size'] for r in results_list])),
                }
                
                dm = all_results[inst_name][alg_name]['distance_mean']
                ds = all_results[inst_name][alg_name]['distance_std']
                hm = all_results[inst_name][alg_name]['hypervolume_mean']
                bm = all_results[inst_name][alg_name]['balance_mean']
                print(f'  {alg_name:15s}: Dist={dm:8.1f}+/-{ds:6.1f}  HV={hm:.3e}  Bal={bm:.4f}')
    
    os.makedirs('experiments_output_v2', exist_ok=True)
    with open('experiments_output_v2/full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print('\n' + '=' * 70)
    print('EXPERIMENTS COMPLETE')
    print('=' * 70)
    
    return all_results


if __name__ == "__main__":
    results = run_experiments()
