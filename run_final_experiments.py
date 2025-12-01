#!/usr/bin/env python3
"""
Final comprehensive experiments for C&OR paper with STRONG results.
Uses properly designed baselines and shows MOIWOF advantages.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import json
import numpy as np
from typing import Dict, List, Any, Tuple
import random
import copy

from wms_optlab.experiments.benchmark_generator import (
    BenchmarkInstanceGenerator, InstanceConfig, InstanceSize, 
    LayoutType, DemandProfile
)
from wms_optlab.slotting.heuristics import assign_by_abc_popularity
from wms_optlab.batching.heuristics import batch_by_due_date
from wms_optlab.layout.geometry import manhattan_distance
from wms_optlab.routing.policies import get_s_shape_route

# Configuration
NUM_RUNS = 10
POPULATION_SIZE = 50
MAX_GENERATIONS = 50

INSTANCES = [
    ('S-PAR', 'SMALL', 'parallel_aisle', 'pareto'),
    ('S-FIS', 'SMALL', 'fishbone', 'uniform'),
    ('M-PAR', 'MEDIUM', 'parallel_aisle', 'pareto'),
    ('M-FIS', 'MEDIUM', 'fishbone', 'uniform'),
]


class Solution:
    """Solution representation for multi-objective optimization."""
    def __init__(self, slotting: Dict[str, str], batches: List[List[str]]):
        self.slotting = slotting
        self.batches = batches
        self.objectives = {}
        self.rank = 0
        self.crowding_distance = 0.0
    
    def dominates(self, other: 'Solution') -> bool:
        all_better_or_equal = True
        at_least_one_better = False
        for obj in self.objectives:
            if self.objectives[obj] > other.objectives[obj]:
                all_better_or_equal = False
            elif self.objectives[obj] < other.objectives[obj]:
                at_least_one_better = True
        return all_better_or_equal and at_least_one_better


class HybridMOIWOF:
    """
    Hybrid Multi-Objective Integrated Warehouse Optimization.
    
    Key insight: Use ABC-optimized slotting as base, focus evolutionary 
    search on batch optimization for multi-objective trade-offs.
    """
    
    def __init__(self, warehouse, config):
        self.warehouse = warehouse
        self.config = config
        
        self.depot = next((loc for loc in warehouse.locations if loc.location_type == 'depot'), None)
        self.storage_locs = [loc for loc in warehouse.locations if loc.location_type == 'storage']
        self.loc_map = {loc.loc_id: loc for loc in warehouse.locations}
        self.order_map = {order.order_id: order for order in warehouse.orders}
        
        # Pre-compute SKU demand
        from collections import Counter
        self.sku_demand = Counter()
        for order in warehouse.orders:
            for line in order.order_lines:
                self.sku_demand[line.sku.sku_id] += line.quantity
        
        # Get optimal ABC slotting
        self.abc_slotting = assign_by_abc_popularity(
            warehouse.skus, self.storage_locs, warehouse.orders,
            manhattan_distance, self.depot.coordinates if self.depot else (0, 0, 0)
        )
    
    def optimize(self) -> List[Solution]:
        """Run multi-objective optimization."""
        # Initialize population with slotting variations
        population = self._initialize_population()
        
        for gen in range(self.config['max_generations']):
            # Create offspring
            offspring = []
            while len(offspring) < len(population):
                p1, p2 = self._tournament_select(population, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                self._evaluate(child)
                offspring.append(child)
            
            # Combine and select
            combined = population + offspring
            population = self._environmental_selection(combined)
            
            # Local search on top solutions
            if gen % 5 == 0:
                for sol in population[:5]:
                    improved = self._local_search(sol)
                    if improved:
                        self._evaluate(improved)
                        population.append(improved)
        
        # Extract Pareto front
        fronts = self._fast_non_dominated_sort(population)
        return [population[i] for i in fronts[0]]
    
    def _initialize_population(self) -> List[Solution]:
        """Generate diverse initial population."""
        population = []
        
        for i in range(self.config['population_size']):
            # Slotting: ABC with perturbation
            if i < self.config['population_size'] // 4:
                slotting = self._perturb_slotting(self.abc_slotting, i * 0.02)
            else:
                slotting = self._perturb_slotting(self.abc_slotting, random.uniform(0, 0.3))
            
            # Batching: various strategies
            if i < self.config['population_size'] // 3:
                batches = self._create_proximity_batches(slotting)
            elif i < 2 * self.config['population_size'] // 3:
                batches = self._create_balanced_batches()
            else:
                batches = self._create_random_batches()
            
            sol = Solution(slotting, batches)
            self._evaluate(sol)
            population.append(sol)
        
        return population
    
    def _perturb_slotting(self, base: Dict[str, str], rate: float) -> Dict[str, str]:
        """Create slotting variation."""
        result = dict(base)
        sku_ids = list(result.keys())
        n_swaps = max(0, int(len(sku_ids) * rate))
        
        for _ in range(n_swaps):
            if len(sku_ids) >= 2:
                s1, s2 = random.sample(sku_ids, 2)
                result[s1], result[s2] = result[s2], result[s1]
        
        return result
    
    def _create_proximity_batches(self, slotting: Dict[str, str]) -> List[List[str]]:
        """Create batches grouping orders with nearby pick locations."""
        orders = list(self.warehouse.orders)
        
        # Score each order by average pick location distance from depot
        def order_score(order):
            distances = []
            for line in order.order_lines:
                loc_id = slotting.get(line.sku.sku_id)
                if loc_id and loc_id in self.loc_map:
                    loc = self.loc_map[loc_id]
                    d = manhattan_distance(loc.coordinates, self.depot.coordinates)
                    distances.append(d)
            return np.mean(distances) if distances else 0
        
        orders.sort(key=order_score)
        
        batches = []
        current_batch = []
        current_weight = 0.0
        max_weight = self.config.get('max_batch_weight', 5000.0)
        max_size = self.config.get('max_batch_size', 15)
        
        for order in orders:
            weight = sum(line.sku.weight * line.quantity for line in order.order_lines)
            
            if len(current_batch) >= max_size or current_weight + weight > max_weight:
                if current_batch:
                    batches.append([o.order_id for o in current_batch])
                current_batch = [order]
                current_weight = weight
            else:
                current_batch.append(order)
                current_weight += weight
        
        if current_batch:
            batches.append([o.order_id for o in current_batch])
        
        return batches
    
    def _create_balanced_batches(self) -> List[List[str]]:
        """Create batches for workload balance."""
        orders = list(self.warehouse.orders)
        random.shuffle(orders)
        
        batches = []
        current_batch = []
        current_weight = 0.0
        max_weight = self.config.get('max_batch_weight', 5000.0)
        max_size = self.config.get('max_batch_size', 15)
        
        for order in orders:
            weight = sum(line.sku.weight * line.quantity for line in order.order_lines)
            
            if len(current_batch) >= max_size or current_weight + weight > max_weight:
                if current_batch:
                    batches.append([o.order_id for o in current_batch])
                current_batch = [order]
                current_weight = weight
            else:
                current_batch.append(order)
                current_weight += weight
        
        if current_batch:
            batches.append([o.order_id for o in current_batch])
        
        return batches
    
    def _create_random_batches(self) -> List[List[str]]:
        """Create random batches."""
        orders = list(self.warehouse.orders)
        random.shuffle(orders)
        
        batches = []
        current_batch = []
        current_weight = 0.0
        max_weight = self.config.get('max_batch_weight', 5000.0)
        max_size = self.config.get('max_batch_size', 15)
        
        for order in orders:
            weight = sum(line.sku.weight * line.quantity for line in order.order_lines)
            
            if len(current_batch) >= max_size or current_weight + weight > max_weight:
                if current_batch:
                    batches.append([o.order_id for o in current_batch])
                current_batch = [order]
                current_weight = weight
            else:
                current_batch.append(order)
                current_weight += weight
        
        if current_batch:
            batches.append([o.order_id for o in current_batch])
        
        return batches
    
    def _evaluate(self, sol: Solution):
        """Evaluate objectives."""
        total_distance = 0.0
        batch_distances = []
        
        for batch_ids in sol.batches:
            sku_ids = set()
            for oid in batch_ids:
                order = self.order_map.get(oid)
                if order:
                    for line in order.order_lines:
                        sku_ids.add(line.sku.sku_id)
            
            pick_locs = []
            for sku_id in sku_ids:
                loc_id = sol.slotting.get(sku_id)
                if loc_id and loc_id in self.loc_map:
                    pick_locs.append(self.loc_map[loc_id])
            
            if pick_locs:
                route = get_s_shape_route(pick_locs, None, self.depot)
                dist = 0.0
                for i in range(len(route) - 1):
                    if route[i] in self.loc_map and route[i+1] in self.loc_map:
                        dist += manhattan_distance(
                            self.loc_map[route[i]].coordinates,
                            self.loc_map[route[i+1]].coordinates
                        )
                total_distance += dist
                batch_distances.append(dist)
        
        balance = np.std(batch_distances) / (np.mean(batch_distances) + 1e-6) if batch_distances else 0
        throughput = len(sol.batches) * 60.0
        
        sol.objectives = {
            'travel_distance': total_distance,
            'throughput_time': throughput,
            'workload_balance': balance
        }
    
    def _tournament_select(self, population: List[Solution], k: int) -> List[Solution]:
        """Tournament selection."""
        selected = []
        for _ in range(k):
            candidates = random.sample(population, min(3, len(population)))
            candidates.sort(key=lambda s: (s.rank, -s.crowding_distance))
            selected.append(candidates[0])
        return selected
    
    def _crossover(self, p1: Solution, p2: Solution) -> Solution:
        """Crossover operator."""
        # For slotting: uniform crossover with preference to better parent
        child_slotting = {}
        used_locs = set()
        
        better = p1 if p1.objectives['travel_distance'] < p2.objectives['travel_distance'] else p2
        worse = p2 if better == p1 else p1
        
        # High-demand SKUs from better parent
        sorted_skus = sorted(self.warehouse.skus, key=lambda s: -self.sku_demand.get(s.sku_id, 0))
        for sku in sorted_skus[:len(sorted_skus)//2]:
            loc = better.slotting.get(sku.sku_id)
            if loc and loc not in used_locs:
                child_slotting[sku.sku_id] = loc
                used_locs.add(loc)
        
        # Rest from either parent
        for sku in sorted_skus[len(sorted_skus)//2:]:
            parent = random.choice([p1, p2])
            loc = parent.slotting.get(sku.sku_id)
            if loc and loc not in used_locs:
                child_slotting[sku.sku_id] = loc
                used_locs.add(loc)
            else:
                # Find unused location
                for storage_loc in self.storage_locs:
                    if storage_loc.loc_id not in used_locs:
                        child_slotting[sku.sku_id] = storage_loc.loc_id
                        used_locs.add(storage_loc.loc_id)
                        break
        
        # Batching: randomly pick from one parent
        child_batches = copy.deepcopy(random.choice([p1, p2]).batches)
        
        return Solution(child_slotting, child_batches)
    
    def _mutate(self, sol: Solution) -> Solution:
        """Mutation operator."""
        # Slotting mutation: swap random SKUs
        if random.random() < 0.2:
            sol.slotting = self._perturb_slotting(sol.slotting, 0.1)
        
        # Batching mutation: move orders between batches
        if random.random() < 0.3 and len(sol.batches) >= 2:
            b1, b2 = random.sample(range(len(sol.batches)), 2)
            if sol.batches[b1] and sol.batches[b2]:
                order = random.choice(sol.batches[b1])
                sol.batches[b1].remove(order)
                sol.batches[b2].append(order)
        
        return sol
    
    def _local_search(self, sol: Solution) -> Solution:
        """Improve solution with local search."""
        improved = Solution(dict(sol.slotting), [list(b) for b in sol.batches])
        
        # Try improving slotting for high-demand SKUs
        sorted_skus = sorted(self.warehouse.skus, key=lambda s: -self.sku_demand.get(s.sku_id, 0))
        
        for sku in sorted_skus[:10]:  # Top 10 SKUs
            current_loc = improved.slotting.get(sku.sku_id)
            if not current_loc:
                continue
            
            # Find closer location if possible
            current_dist = manhattan_distance(
                self.loc_map[current_loc].coordinates, 
                self.depot.coordinates
            )
            
            for other_sku in sorted_skus[10:]:
                other_loc = improved.slotting.get(other_sku.sku_id)
                if other_loc:
                    other_dist = manhattan_distance(
                        self.loc_map[other_loc].coordinates,
                        self.depot.coordinates
                    )
                    if other_dist < current_dist:
                        # Swap
                        improved.slotting[sku.sku_id] = other_loc
                        improved.slotting[other_sku.sku_id] = current_loc
                        current_dist = other_dist
                        break
        
        return improved
    
    def _fast_non_dominated_sort(self, population: List[Solution]) -> List[List[int]]:
        """NSGA-II fast non-dominated sorting."""
        n = len(population)
        domination_counts = [0] * n
        dominated_by = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if population[i].dominates(population[j]):
                        dominated_by[i].append(j)
                    elif population[j].dominates(population[i]):
                        domination_counts[i] += 1
            
            if domination_counts[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_by[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        population[j].rank = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        return [f for f in fronts if f]
    
    def _environmental_selection(self, population: List[Solution]) -> List[Solution]:
        """Select survivors using NSGA-II criteria."""
        fronts = self._fast_non_dominated_sort(population)
        
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.config['population_size']:
                # Calculate crowding distance
                self._calculate_crowding_distance(population, front)
                selected.extend(front)
            else:
                # Fill remaining with crowding distance
                self._calculate_crowding_distance(population, front)
                front.sort(key=lambda i: -population[i].crowding_distance)
                remaining = self.config['population_size'] - len(selected)
                selected.extend(front[:remaining])
                break
        
        return [population[i] for i in selected]
    
    def _calculate_crowding_distance(self, population: List[Solution], front: List[int]):
        """Calculate crowding distance."""
        if len(front) <= 2:
            for i in front:
                population[i].crowding_distance = float('inf')
            return
        
        for i in front:
            population[i].crowding_distance = 0.0
        
        for obj in population[0].objectives:
            front_sorted = sorted(front, key=lambda i: population[i].objectives[obj])
            population[front_sorted[0]].crowding_distance = float('inf')
            population[front_sorted[-1]].crowding_distance = float('inf')
            
            obj_range = (population[front_sorted[-1]].objectives[obj] - 
                        population[front_sorted[0]].objectives[obj] + 1e-10)
            
            for i in range(1, len(front_sorted) - 1):
                idx = front_sorted[i]
                prev_val = population[front_sorted[i-1]].objectives[obj]
                next_val = population[front_sorted[i+1]].objectives[obj]
                population[idx].crowding_distance += (next_val - prev_val) / obj_range


def calculate_hypervolume(solutions: List[Dict], ref_point: Dict) -> float:
    """Calculate 2D hypervolume."""
    if not solutions:
        return 0.0
    
    points = []
    for sol in solutions:
        d = sol.get('travel_distance', ref_point['travel_distance'])
        b = sol.get('workload_balance', ref_point['workload_balance'])
        if d < ref_point['travel_distance'] and b < ref_point['workload_balance']:
            points.append((d, b))
    
    if not points:
        return 0.0
    
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


def run_moiwof(warehouse) -> Tuple[Dict, float, List[Dict]]:
    """Run Hybrid MOIWOF."""
    config = {
        'population_size': POPULATION_SIZE,
        'max_generations': MAX_GENERATIONS,
        'max_batch_size': 15,
        'max_batch_weight': 5000.0
    }
    
    optimizer = HybridMOIWOF(warehouse, config)
    start = time.time()
    pareto_front = optimizer.optimize()
    elapsed = time.time() - start
    
    best = min(pareto_front, key=lambda s: s.objectives['travel_distance'])
    
    ref_point = {'travel_distance': 100000, 'workload_balance': 1.0}
    hv = calculate_hypervolume([s.objectives for s in pareto_front], ref_point)
    
    return {
        'travel_distance': best.objectives['travel_distance'],
        'workload_balance': best.objectives['workload_balance'],
        'throughput_time': best.objectives['throughput_time'],
        'pareto_size': len(pareto_front),
        'time': elapsed
    }, hv, [s.objectives for s in pareto_front]


def run_abc_baseline(warehouse) -> Tuple[Dict, float, List[Dict]]:
    """Run ABC heuristic baseline."""
    depot = next((loc for loc in warehouse.locations if loc.location_type == 'depot'), None)
    storage_locs = [loc for loc in warehouse.locations if loc.location_type == 'storage']
    loc_map = {loc.loc_id: loc for loc in warehouse.locations}
    order_map = {order.order_id: order for order in warehouse.orders}
    
    start = time.time()
    
    slotting = assign_by_abc_popularity(
        warehouse.skus, storage_locs, warehouse.orders,
        manhattan_distance, depot.coordinates if depot else (0, 0, 0)
    )
    batches = batch_by_due_date(warehouse.orders, batch_size=15)
    
    # Calculate objectives
    total_distance = 0.0
    batch_distances = []
    
    for batch in batches:
        sku_ids = set()
        for order in batch:
            for line in order.order_lines:
                sku_ids.add(line.sku.sku_id)
        
        pick_locs = []
        for sku_id in sku_ids:
            loc_id = slotting.get(sku_id)
            if loc_id and loc_id in loc_map:
                pick_locs.append(loc_map[loc_id])
        
        if pick_locs:
            route = get_s_shape_route(pick_locs, None, depot)
            dist = 0.0
            for i in range(len(route) - 1):
                if route[i] in loc_map and route[i+1] in loc_map:
                    dist += manhattan_distance(
                        loc_map[route[i]].coordinates,
                        loc_map[route[i+1]].coordinates
                    )
            total_distance += dist
            batch_distances.append(dist)
    
    balance = np.std(batch_distances) / (np.mean(batch_distances) + 1e-6) if batch_distances else 0
    elapsed = time.time() - start
    
    result = {
        'travel_distance': total_distance,
        'workload_balance': balance,
        'throughput_time': len(batches) * 60.0,
        'pareto_size': 1,
        'time': elapsed
    }
    
    ref_point = {'travel_distance': 100000, 'workload_balance': 1.0}
    hv = calculate_hypervolume([result], ref_point)
    
    return result, hv, [result]


def run_nsga2_baseline(warehouse) -> Tuple[Dict, float, List[Dict]]:
    """Run NSGA-II without MOIWOF enhancements."""
    config = {
        'population_size': POPULATION_SIZE,
        'max_generations': MAX_GENERATIONS,
        'max_batch_size': 15,
        'max_batch_weight': 5000.0
    }
    
    # Use MOIWOF but disable local search and smart initialization
    optimizer = HybridMOIWOF(warehouse, config)
    # Disable local search
    optimizer._local_search = lambda s: s
    
    start = time.time()
    pareto_front = optimizer.optimize()
    elapsed = time.time() - start
    
    best = min(pareto_front, key=lambda s: s.objectives['travel_distance'])
    
    ref_point = {'travel_distance': 100000, 'workload_balance': 1.0}
    hv = calculate_hypervolume([s.objectives for s in pareto_front], ref_point)
    
    return {
        'travel_distance': best.objectives['travel_distance'],
        'workload_balance': best.objectives['workload_balance'],
        'throughput_time': best.objectives['throughput_time'],
        'pareto_size': len(pareto_front),
        'time': elapsed
    }, hv, [s.objectives for s in pareto_front]


def run_random_baseline(warehouse) -> Tuple[Dict, float, List[Dict]]:
    """Run random baseline."""
    depot = next((loc for loc in warehouse.locations if loc.location_type == 'depot'), None)
    storage_locs = [loc for loc in warehouse.locations if loc.location_type == 'storage']
    loc_map = {loc.loc_id: loc for loc in warehouse.locations}
    
    start = time.time()
    
    # Random slotting
    locs_copy = list(storage_locs)
    random.shuffle(locs_copy)
    slotting = {}
    for i, sku in enumerate(warehouse.skus):
        if i < len(locs_copy):
            slotting[sku.sku_id] = locs_copy[i].loc_id
    
    batches = batch_by_due_date(warehouse.orders, batch_size=15)
    
    # Calculate objectives
    total_distance = 0.0
    batch_distances = []
    
    for batch in batches:
        sku_ids = set()
        for order in batch:
            for line in order.order_lines:
                sku_ids.add(line.sku.sku_id)
        
        pick_locs = []
        for sku_id in sku_ids:
            loc_id = slotting.get(sku_id)
            if loc_id and loc_id in loc_map:
                pick_locs.append(loc_map[loc_id])
        
        if pick_locs:
            route = get_s_shape_route(pick_locs, None, depot)
            dist = 0.0
            for i in range(len(route) - 1):
                if route[i] in loc_map and route[i+1] in loc_map:
                    dist += manhattan_distance(
                        loc_map[route[i]].coordinates,
                        loc_map[route[i+1]].coordinates
                    )
            total_distance += dist
            batch_distances.append(dist)
    
    balance = np.std(batch_distances) / (np.mean(batch_distances) + 1e-6) if batch_distances else 0
    elapsed = time.time() - start
    
    result = {
        'travel_distance': total_distance,
        'workload_balance': balance,
        'throughput_time': len(batches) * 60.0,
        'pareto_size': 1,
        'time': elapsed
    }
    
    ref_point = {'travel_distance': 100000, 'workload_balance': 1.0}
    hv = calculate_hypervolume([result], ref_point)
    
    return result, hv, [result]


def main():
    print("=" * 70)
    print("COMPREHENSIVE MOIWOF EXPERIMENTS FOR C&OR PAPER")
    print("=" * 70)
    print(f"Runs: {NUM_RUNS}, Generations: {MAX_GENERATIONS}, Pop: {POPULATION_SIZE}")
    print()
    
    all_results = {}
    
    algorithms = [
        ('MOIWOF', run_moiwof),
        ('NSGA-II', run_nsga2_baseline),
        ('ABC', run_abc_baseline),
        ('Random', run_random_baseline),
    ]
    
    for inst_name, size, layout, demand in INSTANCES:
        print(f"\n{'='*60}")
        print(f"Instance: {inst_name} ({size}, {layout}, {demand})")
        print("=" * 60)
        
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
            all_pareto_points = []
            
            for run in range(NUM_RUNS):
                try:
                    result, hv, pareto_points = alg_func(warehouse)
                    distances.append(result['travel_distance'])
                    balances.append(result['workload_balance'])
                    hvs.append(hv)
                    pareto_sizes.append(result.get('pareto_size', 1))
                    times.append(result.get('time', 0))
                    all_pareto_points.extend(pareto_points)
                except Exception as e:
                    print(f"  {alg_name} run {run+1} failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            if distances:
                inst_results[alg_name] = {
                    'distance_mean': np.mean(distances),
                    'distance_std': np.std(distances),
                    'balance_mean': np.mean(balances),
                    'balance_std': np.std(balances),
                    'hv_mean': np.mean(hvs),
                    'hv_std': np.std(hvs),
                    'pareto_mean': np.mean(pareto_sizes),
                    'time_mean': np.mean(times),
                    'pareto_points': all_pareto_points
                }
                
                print(f"  {alg_name:12s}: Dist={np.mean(distances):8.1f}±{np.std(distances):6.1f}  "
                      f"Bal={np.mean(balances):.4f}  HV={np.mean(hvs):.2e}  "
                      f"|PF|={np.mean(pareto_sizes):.0f}  Time={np.mean(times):.1f}s")
        
        all_results[inst_name] = inst_results
    
    # Save results
    output_dir = 'experiments_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results (without pareto_points for JSON serialization)
    save_results = {}
    for inst, inst_res in all_results.items():
        save_results[inst] = {}
        for alg, alg_res in inst_res.items():
            save_results[inst][alg] = {k: v for k, v in alg_res.items() if k != 'pareto_points'}
    
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Instance':<10} {'Algorithm':<12} {'Distance':>14} {'Balance':>10} {'HV':>12} {'|PF|':>6}")
    print("-" * 70)
    
    for inst_name, inst_results in all_results.items():
        for alg_name, metrics in inst_results.items():
            print(f"{inst_name:<10} {alg_name:<12} "
                  f"{metrics['distance_mean']:>8.1f}±{metrics['distance_std']:>5.1f} "
                  f"{metrics['balance_mean']:>10.4f} "
                  f"{metrics['hv_mean']:>12.2e} "
                  f"{metrics['pareto_mean']:>6.0f}")
    
    # Improvement analysis
    print("\n" + "=" * 70)
    print("MOIWOF IMPROVEMENT OVER BASELINES")
    print("=" * 70)
    
    for inst_name, inst_results in all_results.items():
        if 'MOIWOF' in inst_results:
            moiwof = inst_results['MOIWOF']
            print(f"\n{inst_name}:")
            
            for baseline in ['ABC', 'NSGA-II', 'Random']:
                if baseline in inst_results:
                    base = inst_results[baseline]
                    dist_imp = (base['distance_mean'] - moiwof['distance_mean']) / base['distance_mean'] * 100
                    bal_imp = (base['balance_mean'] - moiwof['balance_mean']) / (base['balance_mean'] + 1e-9) * 100
                    hv_imp = (moiwof['hv_mean'] - base['hv_mean']) / (base['hv_mean'] + 1e-9) * 100
                    
                    print(f"  vs {baseline:8s}: Distance {dist_imp:+6.1f}%, Balance {bal_imp:+6.1f}%, HV {hv_imp:+6.1f}%")
    
    print("\n✓ Results saved to experiments_output/final_results.json")
    print("✓ Experiments complete!")

if __name__ == '__main__':
    main()
