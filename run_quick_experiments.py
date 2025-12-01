#!/usr/bin/env python3
"""
Quick experiments for C&OR paper - minimal but complete runs.
Optimized to complete within 2-3 minutes while providing publishable results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import random

from wms_optlab.experiments.benchmark_generator import (
    BenchmarkInstanceGenerator, InstanceConfig, InstanceSize, 
    LayoutType, DemandProfile
)
from wms_optlab.experiments.moiwof_v2 import MOIWOFv2, MOIWOFConfig
from wms_optlab.slotting.heuristics import assign_by_abc_popularity
from wms_optlab.batching.heuristics import batch_by_due_date
from wms_optlab.layout.geometry import manhattan_distance

# Configuration for quick but valid experiments
NUM_RUNS = 5  # Statistically meaningful
MAX_GENERATIONS = 20  # Enough to converge for small instances
POPULATION_SIZE = 30  # Smaller population

# Two instance sizes for quick validation
INSTANCES = [
    ('S-PAR', 'SMALL', 'parallel_aisle', 'pareto'),  # 75 SKUs, 100 locs
    ('S-FIS', 'SMALL', 'fishbone', 'uniform'),        # 75 SKUs, 100 locs
    ('M-PAR', 'MEDIUM', 'parallel_aisle', 'pareto'), # 300 SKUs, 400 locs
]

def calculate_metrics(warehouse, slotting_plan, batches) -> Dict[str, float]:
    """Calculate optimization objectives."""
    from wms_optlab.routing.policies import get_s_shape_route
    
    loc_map = {loc.loc_id: loc for loc in warehouse.locations}
    depot = next((loc for loc in warehouse.locations if loc.location_type == 'depot'), None)
    if depot is None:
        depot = warehouse.locations[0]
    
    total_distance = 0.0
    batch_distances = []
    
    for batch in batches:
        sku_ids = set()
        for order in batch:
            for line in order.order_lines:
                sku_ids.add(line.sku.sku_id)
        
        pick_locs = []
        for sku_id in sku_ids:
            if sku_id in slotting_plan:
                loc_id = slotting_plan[sku_id]
                if loc_id in loc_map:
                    pick_locs.append(loc_map[loc_id])
        
        if pick_locs:
            route = get_s_shape_route(pick_locs, None, depot)
            dist = 0.0
            for i in range(len(route) - 1):
                if route[i] in loc_map and route[i+1] in loc_map:
                    dist += manhattan_distance(loc_map[route[i]].coordinates, 
                                              loc_map[route[i+1]].coordinates)
            total_distance += dist
            batch_distances.append(dist)
    
    balance = np.std(batch_distances) / (np.mean(batch_distances) + 1e-6) if batch_distances else 0.0
    throughput = len(batches) * 60.0  # Simplified
    
    return {
        'travel_distance': total_distance,
        'throughput_time': throughput,
        'workload_balance': balance
    }

def calculate_hypervolume(solutions: List[Dict], ref_point: Dict) -> float:
    """Simple 2D hypervolume for distance and balance."""
    if not solutions:
        return 0.0
    
    # Normalize and compute hypervolume contribution
    points = []
    for sol in solutions:
        d = sol.get('travel_distance', ref_point['travel_distance'])
        b = sol.get('workload_balance', ref_point['workload_balance'])
        if d < ref_point['travel_distance'] and b < ref_point['workload_balance']:
            points.append((d, b))
    
    if not points:
        return 0.0
    
    # Sort by distance
    points.sort(key=lambda p: p[0])
    
    hv = 0.0
    prev_balance = ref_point['workload_balance']
    for d, b in points:
        width = ref_point['travel_distance'] - d
        height = prev_balance - b
        if height > 0 and width > 0:
            hv += width * height
        prev_balance = min(prev_balance, b)
    
    return hv

def run_moiwof(warehouse, config_name: str, enable_ads: bool = True, enable_ccl: bool = True) -> Tuple[Dict, float]:
    """Run MOIWOF variant."""
    config = MOIWOFConfig(
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        enable_ads=enable_ads,
        enable_ccl=enable_ccl,
        local_search_probability=0.2,
    )
    
    optimizer = MOIWOFv2(warehouse, config)
    start = time.time()
    pareto_front, history = optimizer.run()  # Use run() instead of optimize()
    elapsed = time.time() - start
    
    # Get best solution by distance
    best = min(pareto_front, key=lambda x: x.objectives['travel_distance'])
    
    return {
        'travel_distance': best.objectives['travel_distance'],
        'workload_balance': best.objectives['workload_balance'],
        'throughput_time': best.objectives['throughput_time'],
        'pareto_size': len(pareto_front),
        'time': elapsed
    }, calculate_hypervolume([s.objectives for s in pareto_front], 
                             {'travel_distance': 50000, 'workload_balance': 1.0})

def run_abc_baseline(warehouse) -> Tuple[Dict, float]:
    """Run ABC heuristic baseline."""
    start = time.time()
    
    # Get storage locations only
    storage_locs = [loc for loc in warehouse.locations if loc.location_type == 'storage']
    depot = next((loc for loc in warehouse.locations if loc.location_type == 'depot'), None)
    depot_coords = depot.coordinates if depot else (0, 0, 0)
    
    slotting = assign_by_abc_popularity(
        warehouse.skus, 
        storage_locs, 
        warehouse.orders,
        manhattan_distance,
        depot_coords
    )
    batches = batch_by_due_date(warehouse.orders, batch_size=20)
    elapsed = time.time() - start
    
    metrics = calculate_metrics(warehouse, slotting, batches)
    metrics['time'] = elapsed
    metrics['pareto_size'] = 1
    
    hv = calculate_hypervolume([metrics], {'travel_distance': 50000, 'workload_balance': 1.0})
    return metrics, hv

def run_random_baseline(warehouse) -> Tuple[Dict, float]:
    """Run random assignment baseline."""
    start = time.time()
    
    storage_locs = [loc for loc in warehouse.locations if loc.location_type == 'storage']
    random.shuffle(storage_locs)
    
    slotting = {}
    for i, sku in enumerate(warehouse.skus):
        if i < len(storage_locs):
            slotting[sku.sku_id] = storage_locs[i].loc_id
    
    batches = batch_by_due_date(warehouse.orders, batch_size=20)
    elapsed = time.time() - start
    
    metrics = calculate_metrics(warehouse, slotting, batches)
    metrics['time'] = elapsed
    metrics['pareto_size'] = 1
    
    hv = calculate_hypervolume([metrics], {'travel_distance': 50000, 'workload_balance': 1.0})
    return metrics, hv

def run_nsga2_simplified(warehouse) -> Tuple[Dict, float]:
    """Simplified NSGA-II without MOIWOF enhancements."""
    config = MOIWOFConfig(
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        enable_ads=False,
        enable_ccl=False,
        local_search_probability=0.0,
    )
    
    optimizer = MOIWOFv2(warehouse, config)
    start = time.time()
    pareto_front, history = optimizer.run()  # Use run() instead of optimize()
    elapsed = time.time() - start
    
    best = min(pareto_front, key=lambda x: x.objectives['travel_distance'])
    
    return {
        'travel_distance': best.objectives['travel_distance'],
        'workload_balance': best.objectives['workload_balance'],
        'throughput_time': best.objectives['throughput_time'],
        'pareto_size': len(pareto_front),
        'time': elapsed
    }, calculate_hypervolume([s.objectives for s in pareto_front],
                             {'travel_distance': 50000, 'workload_balance': 1.0})

def main():
    print("=" * 70)
    print("MOIWOF QUICK EXPERIMENTS FOR C&OR PAPER")
    print("=" * 70)
    print(f"Runs: {NUM_RUNS}, Generations: {MAX_GENERATIONS}, Pop: {POPULATION_SIZE}")
    print()
    
    all_results = {}
    
    algorithms = [
        ('MOIWOF', lambda w: run_moiwof(w, 'full', True, True)),
        ('MOIWOF-NoADS', lambda w: run_moiwof(w, 'noads', False, True)),
        ('NSGA-II', run_nsga2_simplified),
        ('ABC', run_abc_baseline),
        ('Random', run_random_baseline),
    ]
    
    for inst_name, size, layout, demand in INSTANCES:
        print(f"\n{'='*50}")
        print(f"Instance: {inst_name} ({size}, {layout})")
        print("=" * 50)
        
        # Create config and generate
        config = InstanceConfig(
            size=getattr(InstanceSize, size),
            layout_type=getattr(LayoutType, layout.upper()),
            demand_profile=getattr(DemandProfile, demand.upper()),
            random_seed=42
        )
        generator = BenchmarkInstanceGenerator(config)
        instance = generator.generate(inst_name)
        warehouse = instance.warehouse
        
        print(f"SKUs: {len(warehouse.skus)}, Locs: {len(warehouse.locations)}, Orders: {len(warehouse.orders)}")
        
        inst_results = {}
        
        for alg_name, alg_func in algorithms:
            distances = []
            balances = []
            hvs = []
            pareto_sizes = []
            times = []
            
            for run in range(NUM_RUNS):
                try:
                    result, hv = alg_func(warehouse)
                    distances.append(result['travel_distance'])
                    balances.append(result['workload_balance'])
                    hvs.append(hv)
                    pareto_sizes.append(result.get('pareto_size', 1))
                    times.append(result.get('time', 0))
                except Exception as e:
                    print(f"  {alg_name} run {run+1} failed: {e}")
            
            if distances:
                inst_results[alg_name] = {
                    'distance_mean': np.mean(distances),
                    'distance_std': np.std(distances),
                    'balance_mean': np.mean(balances),
                    'balance_std': np.std(balances),
                    'hv_mean': np.mean(hvs),
                    'hv_std': np.std(hvs),
                    'pareto_mean': np.mean(pareto_sizes),
                    'time_mean': np.mean(times)
                }
                
                print(f"  {alg_name:12s}: Dist={np.mean(distances):7.1f}±{np.std(distances):5.1f}  "
                      f"Bal={np.mean(balances):.4f}  HV={np.mean(hvs):.2e}  "
                      f"Pareto={np.mean(pareto_sizes):.0f}  Time={np.mean(times):.1f}s")
        
        all_results[inst_name] = inst_results
    
    # Save results
    output_dir = 'experiments_output'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'quick_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR C&OR PAPER")
    print("=" * 70)
    print(f"{'Instance':<10} {'Algorithm':<12} {'Distance':>12} {'Balance':>10} {'HV':>12}")
    print("-" * 70)
    
    for inst_name, inst_results in all_results.items():
        for alg_name, metrics in inst_results.items():
            print(f"{inst_name:<10} {alg_name:<12} "
                  f"{metrics['distance_mean']:>8.1f}±{metrics['distance_std']:>4.1f} "
                  f"{metrics['balance_mean']:>10.4f} "
                  f"{metrics['hv_mean']:>12.2e}")
    
    # Compute improvement percentages
    print("\n" + "=" * 70)
    print("IMPROVEMENT OF MOIWOF OVER BASELINES")
    print("=" * 70)
    
    for inst_name, inst_results in all_results.items():
        if 'MOIWOF' in inst_results and 'ABC' in inst_results:
            moiwof = inst_results['MOIWOF']
            abc = inst_results['ABC']
            
            dist_imp = (abc['distance_mean'] - moiwof['distance_mean']) / abc['distance_mean'] * 100
            bal_imp = (abc['balance_mean'] - moiwof['balance_mean']) / (abc['balance_mean'] + 1e-9) * 100
            
            print(f"{inst_name}: Distance {dist_imp:+.1f}%, Balance {bal_imp:+.1f}%")
    
    print("\n✓ Results saved to experiments_output/quick_results.json")
    print("✓ Experiments complete!")

if __name__ == '__main__':
    main()
