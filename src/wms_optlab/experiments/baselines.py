"""
Baseline Algorithms for Comparative Experiments.

Implements:
1. NSGA-II (vanilla) - Standard NSGA-II without warehouse-specific enhancements
2. MOEA/D - Multi-Objective Evolutionary Algorithm based on Decomposition
3. Sequential Optimization - Traditional hierarchical approach (Slotting → Batching → Routing)
4. ABC Heuristic - Activity-Based Classification baseline
5. Random - Random assignment baseline
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import copy

from ..data.models import Warehouse, Order
from ..layout.geometry import manhattan_distance
from ..slotting.heuristics import assign_by_abc_popularity
from ..routing.policies import get_s_shape_route
from .moiwof import ParetoSolution, ObjectiveType, MOIWOFConfig


class BaselineAlgorithm:
    """Base class for all baseline algorithms."""
    
    def __init__(self, warehouse: Warehouse, config: MOIWOFConfig):
        self.warehouse = warehouse
        self.config = config
        self.depot = next((loc for loc in warehouse.locations if loc.location_type == 'depot'), None)
        self.graph = warehouse.get_graph()
        self.loc_map = {loc.loc_id: loc for loc in warehouse.locations}
        self.sku_map = {sku.sku_id: sku for sku in warehouse.skus}
        self.order_map = {order.order_id: order for order in warehouse.orders}
        self.storage_locations = [loc for loc in warehouse.locations if loc.location_type == 'storage']
        
        # SKU demand for ABC analysis
        self.sku_demand = Counter()
        for order in warehouse.orders:
            for line in order.order_lines:
                self.sku_demand[line.sku.sku_id] += line.quantity
        
        self.generation_history: List[Dict[str, Any]] = []
    
    def run(self) -> Tuple[List[ParetoSolution], List[Dict[str, Any]]]:
        """Run the algorithm and return Pareto front and history."""
        raise NotImplementedError
    
    def _compute_routes(self, slotting_plan: Dict[str, str], 
                        batches: List[List[str]]) -> Dict[str, List[str]]:
        """Compute routes for each batch."""
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
            
            pick_locations = [self.loc_map[loc_id] for loc_id in pick_loc_ids 
                             if loc_id in self.loc_map]
            
            if pick_locations:
                route = get_s_shape_route(pick_locations, self.graph, self.depot)
            else:
                route = [self.depot.loc_id, self.depot.loc_id]
            
            routes[f"batch_{batch_idx}"] = route
        
        return routes
    
    def _evaluate_objectives(self, slotting_plan: Dict[str, str],
                            batches: List[List[str]],
                            routes: Dict[str, List[str]]) -> Dict[str, float]:
        """Evaluate the three objectives."""
        # Travel Distance
        total_distance = 0.0
        batch_distances = []
        
        for route in routes.values():
            route_dist = 0.0
            for i in range(len(route) - 1):
                loc1 = self.loc_map.get(route[i])
                loc2 = self.loc_map.get(route[i + 1])
                if loc1 and loc2:
                    route_dist += manhattan_distance(loc1.coordinates, loc2.coordinates)
            total_distance += route_dist
            batch_distances.append(route_dist)
        
        # Throughput Time
        batch_times = []
        for route, dist in zip(routes.values(), batch_distances):
            travel_time = dist / 1.0  # velocity = 1
            pick_time = len(set(route)) * 0.5
            batch_times.append(travel_time + pick_time)
        
        picker_workloads = [0.0] * self.config.num_pickers
        for i, time in enumerate(batch_times):
            picker_workloads[i % self.config.num_pickers] += time
        
        throughput_time = max(picker_workloads) if picker_workloads else 0.0
        
        # Workload Balance
        if picker_workloads and np.mean(picker_workloads) > 0:
            workload_balance = np.std(picker_workloads) / np.mean(picker_workloads)
        else:
            workload_balance = 0.0
        
        return {
            ObjectiveType.TRAVEL_DISTANCE.value: total_distance,
            ObjectiveType.THROUGHPUT_TIME.value: throughput_time,
            ObjectiveType.WORKLOAD_BALANCE.value: workload_balance
        }
    
    def _get_non_dominated(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Extract non-dominated solutions."""
        non_dominated = []
        for i, sol in enumerate(solutions):
            dominated = False
            for j, other in enumerate(solutions):
                if i != j and other.dominates(sol):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(sol)
        return non_dominated


class NSGA2Vanilla(BaselineAlgorithm):
    """
    Standard NSGA-II without warehouse-specific enhancements.
    
    This serves as a baseline to show the contribution of
    domain-specific operators in MOIWOF.
    """
    
    def run(self) -> Tuple[List[ParetoSolution], List[Dict[str, Any]]]:
        """Run vanilla NSGA-II."""
        print("Running NSGA-II (vanilla)...")
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Initialize random population
        population = self._initialize_population()
        
        # Initial sorting
        fronts = self._fast_non_dominated_sort(population)
        for front in fronts:
            self._calculate_crowding_distance(population, front)
        
        self.generation_history = []
        
        for gen in range(self.config.max_generations):
            # Generate offspring
            offspring = []
            while len(offspring) < self.config.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                if random.random() < self.config.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))
                
                child = self._mutate(child)
                offspring.append(child)
            
            # Combine and select
            combined = population + offspring
            fronts = self._fast_non_dominated_sort(combined)
            
            new_population = []
            front_idx = 0
            
            while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= self.config.population_size:
                self._calculate_crowding_distance(combined, fronts[front_idx])
                new_population.extend([combined[i] for i in fronts[front_idx]])
                front_idx += 1
            
            if len(new_population) < self.config.population_size and front_idx < len(fronts):
                self._calculate_crowding_distance(combined, fronts[front_idx])
                remaining = sorted(fronts[front_idx],
                                  key=lambda i: -combined[i].crowding_distance)
                for i in remaining:
                    if len(new_population) >= self.config.population_size:
                        break
                    new_population.append(combined[i])
            
            population = new_population
            
            # Record statistics
            best_dist = min(s.objectives[ObjectiveType.TRAVEL_DISTANCE.value] for s in population)
            best_balance = min(s.objectives[ObjectiveType.WORKLOAD_BALANCE.value] for s in population)
            
            self.generation_history.append({
                'generation': gen,
                'best_travel_distance': best_dist,
                'best_workload_balance': best_balance,
                'pareto_front_size': len([s for s in population if s.rank == 0])
            })
            
            if gen % 20 == 0:
                print(f"  Gen {gen}: Dist={best_dist:.1f}")
        
        pareto_front = self._get_non_dominated(population)
        print(f"  Complete. Pareto front: {len(pareto_front)} solutions")
        
        return pareto_front, self.generation_history
    
    def _initialize_population(self) -> List[ParetoSolution]:
        """Initialize with random solutions."""
        population = []
        
        for _ in range(self.config.population_size):
            # Random slotting
            skus = list(self.warehouse.skus)
            locs = list(self.storage_locations)
            random.shuffle(skus)
            random.shuffle(locs)
            
            slotting_plan = {}
            for i, sku in enumerate(skus):
                if i < len(locs):
                    slotting_plan[sku.sku_id] = locs[i].loc_id
            
            # Random batching
            orders = list(self.warehouse.orders)
            random.shuffle(orders)
            batches = []
            for i in range(0, len(orders), self.config.max_batch_size):
                batch = [o.order_id for o in orders[i:i+self.config.max_batch_size]]
                batches.append(batch)
            
            routes = self._compute_routes(slotting_plan, batches)
            objectives = self._evaluate_objectives(slotting_plan, batches, routes)
            
            population.append(ParetoSolution(
                slotting_plan=slotting_plan,
                batches=batches,
                routes=routes,
                objectives=objectives
            ))
        
        return population
    
    def _fast_non_dominated_sort(self, population: List[ParetoSolution]) -> List[List[int]]:
        """NSGA-II fast non-dominated sorting."""
        n = len(population)
        domination_counts = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if population[i].dominates(population[j]):
                        dominated_solutions[i].append(j)
                    elif population[j].dominates(population[i]):
                        domination_counts[i] += 1
            
            if domination_counts[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)
        
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        population[j].rank = current_front + 1
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        return [f for f in fronts if f]
    
    def _calculate_crowding_distance(self, population: List[ParetoSolution], front: List[int]):
        """Calculate crowding distance."""
        if len(front) <= 2:
            for idx in front:
                population[idx].crowding_distance = float('inf')
            return
        
        for idx in front:
            population[idx].crowding_distance = 0.0
        
        objectives = list(population[0].objectives.keys())
        
        for obj in objectives:
            front_sorted = sorted(front, key=lambda i: population[i].objectives[obj])
            
            population[front_sorted[0]].crowding_distance = float('inf')
            population[front_sorted[-1]].crowding_distance = float('inf')
            
            obj_min = population[front_sorted[0]].objectives[obj]
            obj_max = population[front_sorted[-1]].objectives[obj]
            obj_range = obj_max - obj_min + 1e-10
            
            for i in range(1, len(front_sorted) - 1):
                idx = front_sorted[i]
                prev_val = population[front_sorted[i - 1]].objectives[obj]
                next_val = population[front_sorted[i + 1]].objectives[obj]
                population[idx].crowding_distance += (next_val - prev_val) / obj_range
    
    def _tournament_selection(self, population: List[ParetoSolution]) -> ParetoSolution:
        """Binary tournament selection."""
        candidates = random.sample(population, min(2, len(population)))
        return min(candidates, key=lambda s: (s.rank, -s.crowding_distance))
    
    def _crossover(self, parent1: ParetoSolution, parent2: ParetoSolution) -> ParetoSolution:
        """Simple uniform crossover."""
        child_slotting = {}
        used_locations = set()
        
        for sku_id in set(parent1.slotting_plan.keys()) | set(parent2.slotting_plan.keys()):
            if random.random() < 0.5:
                loc = parent1.slotting_plan.get(sku_id)
            else:
                loc = parent2.slotting_plan.get(sku_id)
            
            if loc and loc not in used_locations:
                child_slotting[sku_id] = loc
                used_locations.add(loc)
            else:
                for storage_loc in self.storage_locations:
                    if storage_loc.loc_id not in used_locations:
                        child_slotting[sku_id] = storage_loc.loc_id
                        used_locations.add(storage_loc.loc_id)
                        break
        
        # Simple batch combination
        all_orders = list(set([oid for batch in parent1.batches for oid in batch]))
        random.shuffle(all_orders)
        
        batches = []
        for i in range(0, len(all_orders), self.config.max_batch_size):
            batches.append(all_orders[i:i+self.config.max_batch_size])
        
        routes = self._compute_routes(child_slotting, batches)
        objectives = self._evaluate_objectives(child_slotting, batches, routes)
        
        return ParetoSolution(
            slotting_plan=child_slotting,
            batches=batches,
            routes=routes,
            objectives=objectives,
            generation=max(parent1.generation, parent2.generation) + 1
        )
    
    def _mutate(self, solution: ParetoSolution) -> ParetoSolution:
        """Simple random mutation."""
        new_slotting = dict(solution.slotting_plan)
        new_batches = [list(batch) for batch in solution.batches]
        
        # Random swap mutation
        if random.random() < self.config.mutation_rate:
            sku_ids = list(new_slotting.keys())
            if len(sku_ids) >= 2:
                sku1, sku2 = random.sample(sku_ids, 2)
                new_slotting[sku1], new_slotting[sku2] = new_slotting[sku2], new_slotting[sku1]
        
        # Batch mutation
        if random.random() < self.config.mutation_rate and len(new_batches) >= 2:
            b1, b2 = random.sample(range(len(new_batches)), 2)
            if new_batches[b1] and new_batches[b2]:
                order = random.choice(new_batches[b1])
                new_batches[b1].remove(order)
                new_batches[b2].append(order)
        
        new_batches = [b for b in new_batches if b]
        
        routes = self._compute_routes(new_slotting, new_batches)
        objectives = self._evaluate_objectives(new_slotting, new_batches, routes)
        
        return ParetoSolution(
            slotting_plan=new_slotting,
            batches=new_batches,
            routes=routes,
            objectives=objectives,
            generation=solution.generation + 1
        )


class MOEAD(BaselineAlgorithm):
    """
    MOEA/D - Multi-Objective Evolutionary Algorithm based on Decomposition.
    
    Decomposes the multi-objective problem into scalar subproblems using
    weight vectors and Tchebycheff approach.
    """
    
    def __init__(self, warehouse: Warehouse, config: MOIWOFConfig, 
                 n_neighbors: int = 20):
        super().__init__(warehouse, config)
        self.n_neighbors = n_neighbors
        self.weight_vectors = self._generate_weight_vectors()
        self.neighborhoods = self._compute_neighborhoods()
        self.reference_point: Dict[str, float] = {}  # Ideal point
    
    def _generate_weight_vectors(self) -> List[Tuple[float, float, float]]:
        """Generate uniformly distributed weight vectors for 3 objectives."""
        weights = []
        H = int(np.sqrt(self.config.population_size))  # Divisions
        
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                w1 = i / H
                w2 = j / H
                w3 = k / H
                if w1 + w2 + w3 > 0:
                    weights.append((w1, w2, w3))
        
        # Trim or extend to population size
        while len(weights) < self.config.population_size:
            weights.append((random.random(), random.random(), random.random()))
        
        return weights[:self.config.population_size]
    
    def _compute_neighborhoods(self) -> List[List[int]]:
        """Compute neighborhood structure based on weight vector distances."""
        n = len(self.weight_vectors)
        neighborhoods = []
        
        for i in range(n):
            distances = []
            for j in range(n):
                dist = np.sqrt(sum((self.weight_vectors[i][k] - self.weight_vectors[j][k])**2 
                                   for k in range(3)))
                distances.append((j, dist))
            
            distances.sort(key=lambda x: x[1])
            neighborhood = [idx for idx, _ in distances[:self.n_neighbors]]
            neighborhoods.append(neighborhood)
        
        return neighborhoods
    
    def _tchebycheff(self, objectives: Dict[str, float], 
                     weights: Tuple[float, float, float]) -> float:
        """Tchebycheff scalarization function."""
        obj_list = [
            ObjectiveType.TRAVEL_DISTANCE.value,
            ObjectiveType.THROUGHPUT_TIME.value,
            ObjectiveType.WORKLOAD_BALANCE.value
        ]
        
        max_term = 0.0
        for i, obj in enumerate(obj_list):
            diff = abs(objectives[obj] - self.reference_point.get(obj, 0))
            weighted = weights[i] * diff
            max_term = max(max_term, weighted)
        
        return max_term
    
    def run(self) -> Tuple[List[ParetoSolution], List[Dict[str, Any]]]:
        """Run MOEA/D algorithm."""
        print("Running MOEA/D...")
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Initialize population
        population = self._initialize_population()
        
        # Initialize reference point (ideal)
        self._update_reference_point(population)
        
        # External archive
        archive: List[ParetoSolution] = []
        
        self.generation_history = []
        
        for gen in range(self.config.max_generations):
            for i in range(len(population)):
                # Select from neighborhood
                if random.random() < 0.9:
                    mating_pool = self.neighborhoods[i]
                else:
                    mating_pool = list(range(len(population)))
                
                # Select parents
                p1_idx, p2_idx = random.sample(mating_pool, 2)
                parent1 = population[p1_idx]
                parent2 = population[p2_idx]
                
                # Generate offspring
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                
                # Update reference point
                for obj, val in child.objectives.items():
                    if obj not in self.reference_point or val < self.reference_point[obj]:
                        self.reference_point[obj] = val
                
                # Update neighboring solutions
                for j in self.neighborhoods[i]:
                    child_fitness = self._tchebycheff(child.objectives, self.weight_vectors[j])
                    current_fitness = self._tchebycheff(population[j].objectives, self.weight_vectors[j])
                    
                    if child_fitness < current_fitness:
                        population[j] = child
                
                # Update archive
                archive = self._update_archive(archive, child)
            
            # Record statistics
            best_dist = min(s.objectives[ObjectiveType.TRAVEL_DISTANCE.value] for s in population)
            best_balance = min(s.objectives[ObjectiveType.WORKLOAD_BALANCE.value] for s in population)
            
            self.generation_history.append({
                'generation': gen,
                'best_travel_distance': best_dist,
                'best_workload_balance': best_balance,
                'archive_size': len(archive)
            })
            
            if gen % 20 == 0:
                print(f"  Gen {gen}: Dist={best_dist:.1f}, Archive={len(archive)}")
        
        pareto_front = self._get_non_dominated(archive) if archive else self._get_non_dominated(population)
        print(f"  Complete. Pareto front: {len(pareto_front)} solutions")
        
        return pareto_front, self.generation_history
    
    def _initialize_population(self) -> List[ParetoSolution]:
        """Initialize population with diverse solutions."""
        population = []
        
        # One ABC solution
        abc_slotting = assign_by_abc_popularity(
            self.warehouse.skus, self.warehouse.locations, self.warehouse.orders,
            distance_metric=manhattan_distance,
            depot_location=self.depot.coordinates
        )
        
        for i in range(self.config.population_size):
            if i == 0:
                slotting_plan = abc_slotting
            else:
                # Random with ABC bias
                skus = sorted(self.warehouse.skus, 
                             key=lambda s: -self.sku_demand.get(s.sku_id, 0) + random.gauss(0, i))
                locs = sorted(self.storage_locations,
                             key=lambda l: manhattan_distance(l.coordinates, self.depot.coordinates) + random.gauss(0, i))
                
                slotting_plan = {}
                for j, sku in enumerate(skus):
                    if j < len(locs):
                        slotting_plan[sku.sku_id] = locs[j].loc_id
            
            # Create batches
            orders = list(self.warehouse.orders)
            random.shuffle(orders)
            batches = []
            for j in range(0, len(orders), self.config.max_batch_size):
                batches.append([o.order_id for o in orders[j:j+self.config.max_batch_size]])
            
            routes = self._compute_routes(slotting_plan, batches)
            objectives = self._evaluate_objectives(slotting_plan, batches, routes)
            
            population.append(ParetoSolution(
                slotting_plan=slotting_plan,
                batches=batches,
                routes=routes,
                objectives=objectives
            ))
        
        return population
    
    def _update_reference_point(self, population: List[ParetoSolution]):
        """Update ideal reference point."""
        objectives = [
            ObjectiveType.TRAVEL_DISTANCE.value,
            ObjectiveType.THROUGHPUT_TIME.value,
            ObjectiveType.WORKLOAD_BALANCE.value
        ]
        
        for obj in objectives:
            self.reference_point[obj] = min(s.objectives[obj] for s in population)
    
    def _update_archive(self, archive: List[ParetoSolution], 
                        new_sol: ParetoSolution) -> List[ParetoSolution]:
        """Update external archive with new solution."""
        # Check if new solution is dominated
        dominated = False
        for sol in archive:
            if sol.dominates(new_sol):
                dominated = True
                break
        
        if not dominated:
            # Remove solutions dominated by new solution
            archive = [sol for sol in archive if not new_sol.dominates(sol)]
            archive.append(new_sol)
            
            # Trim if too large
            if len(archive) > self.config.archive_size:
                archive = archive[-self.config.archive_size:]
        
        return archive
    
    def _crossover(self, parent1: ParetoSolution, parent2: ParetoSolution) -> ParetoSolution:
        """Crossover operator."""
        child_slotting = {}
        used_locations = set()
        
        for sku_id in set(parent1.slotting_plan.keys()) | set(parent2.slotting_plan.keys()):
            if random.random() < 0.5:
                loc = parent1.slotting_plan.get(sku_id)
            else:
                loc = parent2.slotting_plan.get(sku_id)
            
            if loc and loc not in used_locations:
                child_slotting[sku_id] = loc
                used_locations.add(loc)
            else:
                for storage_loc in self.storage_locations:
                    if storage_loc.loc_id not in used_locations:
                        child_slotting[sku_id] = storage_loc.loc_id
                        used_locations.add(storage_loc.loc_id)
                        break
        
        all_orders = list(set([oid for batch in parent1.batches for oid in batch]))
        random.shuffle(all_orders)
        
        batches = []
        for i in range(0, len(all_orders), self.config.max_batch_size):
            batches.append(all_orders[i:i+self.config.max_batch_size])
        
        routes = self._compute_routes(child_slotting, batches)
        objectives = self._evaluate_objectives(child_slotting, batches, routes)
        
        return ParetoSolution(
            slotting_plan=child_slotting,
            batches=batches,
            routes=routes,
            objectives=objectives
        )
    
    def _mutate(self, solution: ParetoSolution) -> ParetoSolution:
        """Mutation operator."""
        new_slotting = dict(solution.slotting_plan)
        new_batches = [list(batch) for batch in solution.batches]
        
        if random.random() < self.config.mutation_rate:
            sku_ids = list(new_slotting.keys())
            if len(sku_ids) >= 2:
                sku1, sku2 = random.sample(sku_ids, 2)
                new_slotting[sku1], new_slotting[sku2] = new_slotting[sku2], new_slotting[sku1]
        
        if random.random() < self.config.mutation_rate and len(new_batches) >= 2:
            b1, b2 = random.sample(range(len(new_batches)), 2)
            if new_batches[b1] and new_batches[b2]:
                order = random.choice(new_batches[b1])
                new_batches[b1].remove(order)
                new_batches[b2].append(order)
        
        new_batches = [b for b in new_batches if b]
        
        routes = self._compute_routes(new_slotting, new_batches)
        objectives = self._evaluate_objectives(new_slotting, new_batches, routes)
        
        return ParetoSolution(
            slotting_plan=new_slotting,
            batches=new_batches,
            routes=routes,
            objectives=objectives
        )


class SequentialOptimization(BaselineAlgorithm):
    """
    Traditional Sequential Optimization approach.
    
    Optimizes in sequence: Slotting → Batching → Routing
    without feedback between stages.
    
    This represents the state-of-practice in many warehouses.
    """
    
    def run(self) -> Tuple[List[ParetoSolution], List[Dict[str, Any]]]:
        """Run sequential optimization."""
        print("Running Sequential Optimization...")
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        solutions = []
        
        # Stage 1: Optimize Slotting (multiple strategies)
        slotting_strategies = [
            ('ABC', self._abc_slotting()),
            ('Distance', self._distance_slotting()),
            ('Frequency', self._frequency_slotting()),
            ('Random', self._random_slotting())
        ]
        
        # Stage 2: Optimize Batching for each slotting (multiple strategies)
        batching_strategies = [
            ('FCFS', lambda: self._fcfs_batching()),
            ('Proximity', lambda slotting: self._proximity_batching(slotting)),
            ('Savings', lambda slotting: self._savings_batching(slotting))
        ]
        
        # Combine all strategy combinations
        for slot_name, slotting_plan in slotting_strategies:
            for batch_name, batch_func in batching_strategies:
                try:
                    if batch_name == 'FCFS':
                        batches = batch_func()
                    else:
                        batches = batch_func(slotting_plan)
                    
                    # Stage 3: Compute Routes (S-shape)
                    routes = self._compute_routes(slotting_plan, batches)
                    objectives = self._evaluate_objectives(slotting_plan, batches, routes)
                    
                    solutions.append(ParetoSolution(
                        slotting_plan=slotting_plan,
                        batches=batches,
                        routes=routes,
                        objectives=objectives
                    ))
                except Exception as e:
                    print(f"  Warning: {slot_name}-{batch_name} failed: {e}")
        
        pareto_front = self._get_non_dominated(solutions)
        
        # Simple history (single generation)
        self.generation_history = [{
            'generation': 0,
            'best_travel_distance': min(s.objectives[ObjectiveType.TRAVEL_DISTANCE.value] 
                                        for s in pareto_front),
            'best_workload_balance': min(s.objectives[ObjectiveType.WORKLOAD_BALANCE.value] 
                                         for s in pareto_front),
            'pareto_front_size': len(pareto_front)
        }]
        
        print(f"  Complete. Pareto front: {len(pareto_front)} solutions")
        
        return pareto_front, self.generation_history
    
    def _abc_slotting(self) -> Dict[str, str]:
        """ABC popularity-based slotting."""
        return assign_by_abc_popularity(
            self.warehouse.skus, self.warehouse.locations, self.warehouse.orders,
            distance_metric=manhattan_distance,
            depot_location=self.depot.coordinates
        )
    
    def _distance_slotting(self) -> Dict[str, str]:
        """Distance-only slotting (closest locations first)."""
        sorted_locs = sorted(self.storage_locations,
                            key=lambda l: manhattan_distance(l.coordinates, self.depot.coordinates))
        skus = list(self.warehouse.skus)
        random.shuffle(skus)
        
        plan = {}
        for i, sku in enumerate(skus):
            if i < len(sorted_locs):
                plan[sku.sku_id] = sorted_locs[i].loc_id
        return plan
    
    def _frequency_slotting(self) -> Dict[str, str]:
        """Frequency-based slotting."""
        sorted_skus = sorted(self.warehouse.skus,
                            key=lambda s: -self.sku_demand.get(s.sku_id, 0))
        sorted_locs = sorted(self.storage_locations,
                            key=lambda l: manhattan_distance(l.coordinates, self.depot.coordinates))
        
        plan = {}
        for i, sku in enumerate(sorted_skus):
            if i < len(sorted_locs):
                plan[sku.sku_id] = sorted_locs[i].loc_id
        return plan
    
    def _random_slotting(self) -> Dict[str, str]:
        """Random slotting."""
        skus = list(self.warehouse.skus)
        locs = list(self.storage_locations)
        random.shuffle(skus)
        random.shuffle(locs)
        
        plan = {}
        for i, sku in enumerate(skus):
            if i < len(locs):
                plan[sku.sku_id] = locs[i].loc_id
        return plan
    
    def _fcfs_batching(self) -> List[List[str]]:
        """First-Come-First-Served batching."""
        orders = sorted(self.warehouse.orders, key=lambda o: o.due_date)
        batches = []
        
        for i in range(0, len(orders), self.config.max_batch_size):
            batch = [o.order_id for o in orders[i:i+self.config.max_batch_size]]
            batches.append(batch)
        
        return batches
    
    def _proximity_batching(self, slotting_plan: Dict[str, str]) -> List[List[str]]:
        """Proximity-based batching."""
        # Calculate order centroids
        order_centroids = {}
        for order in self.warehouse.orders:
            locs = []
            for line in order.order_lines:
                loc_id = slotting_plan.get(line.sku.sku_id)
                if loc_id and loc_id in self.loc_map:
                    locs.append(self.loc_map[loc_id].coordinates)
            
            if locs:
                centroid = (np.mean([l[0] for l in locs]),
                           np.mean([l[1] for l in locs]),
                           np.mean([l[2] for l in locs]))
            else:
                centroid = self.depot.coordinates
            order_centroids[order.order_id] = centroid
        
        # Sort by distance to depot
        orders = sorted(self.warehouse.orders,
                       key=lambda o: manhattan_distance(order_centroids[o.order_id], 
                                                       self.depot.coordinates))
        
        batches = []
        for i in range(0, len(orders), self.config.max_batch_size):
            batch = [o.order_id for o in orders[i:i+self.config.max_batch_size]]
            batches.append(batch)
        
        return batches
    
    def _savings_batching(self, slotting_plan: Dict[str, str]) -> List[List[str]]:
        """Savings-based batching (simplified Clarke-Wright)."""
        orders = list(self.warehouse.orders)
        
        # Calculate order centroids
        order_centroids = {}
        for order in orders:
            locs = []
            for line in order.order_lines:
                loc_id = slotting_plan.get(line.sku.sku_id)
                if loc_id and loc_id in self.loc_map:
                    locs.append(self.loc_map[loc_id].coordinates)
            
            if locs:
                centroid = (np.mean([l[0] for l in locs]),
                           np.mean([l[1] for l in locs]),
                           np.mean([l[2] for l in locs]))
            else:
                centroid = self.depot.coordinates
            order_centroids[order.order_id] = centroid
        
        # Calculate savings for order pairs
        savings = []
        for i, o1 in enumerate(orders):
            for o2 in orders[i+1:]:
                d_depot_o1 = manhattan_distance(self.depot.coordinates, 
                                               order_centroids[o1.order_id])
                d_depot_o2 = manhattan_distance(self.depot.coordinates, 
                                               order_centroids[o2.order_id])
                d_o1_o2 = manhattan_distance(order_centroids[o1.order_id],
                                            order_centroids[o2.order_id])
                saving = d_depot_o1 + d_depot_o2 - d_o1_o2
                savings.append((saving, o1.order_id, o2.order_id))
        
        # Sort by savings (descending)
        savings.sort(key=lambda x: -x[0])
        
        # Build batches greedily
        order_to_batch = {}
        batches_dict = {}
        batch_counter = 0
        
        for saving, o1_id, o2_id in savings:
            b1 = order_to_batch.get(o1_id)
            b2 = order_to_batch.get(o2_id)
            
            if b1 is None and b2 is None:
                # Create new batch
                batches_dict[batch_counter] = [o1_id, o2_id]
                order_to_batch[o1_id] = batch_counter
                order_to_batch[o2_id] = batch_counter
                batch_counter += 1
            elif b1 is not None and b2 is None:
                if len(batches_dict[b1]) < self.config.max_batch_size:
                    batches_dict[b1].append(o2_id)
                    order_to_batch[o2_id] = b1
            elif b1 is None and b2 is not None:
                if len(batches_dict[b2]) < self.config.max_batch_size:
                    batches_dict[b2].append(o1_id)
                    order_to_batch[o1_id] = b2
        
        # Add remaining orders
        for order in orders:
            if order.order_id not in order_to_batch:
                batches_dict[batch_counter] = [order.order_id]
                order_to_batch[order.order_id] = batch_counter
                batch_counter += 1
        
        return list(batches_dict.values())


class ABCHeuristic(BaselineAlgorithm):
    """ABC heuristic baseline (single solution)."""
    
    def run(self) -> Tuple[List[ParetoSolution], List[Dict[str, Any]]]:
        """Run ABC heuristic."""
        print("Running ABC Heuristic...")
        
        slotting_plan = assign_by_abc_popularity(
            self.warehouse.skus, self.warehouse.locations, self.warehouse.orders,
            distance_metric=manhattan_distance,
            depot_location=self.depot.coordinates
        )
        
        # Due-date batching
        orders_sorted = sorted(self.warehouse.orders, key=lambda o: o.due_date)
        batches = []
        for i in range(0, len(orders_sorted), self.config.max_batch_size):
            batch = [o.order_id for o in orders_sorted[i:i+self.config.max_batch_size]]
            batches.append(batch)
        
        routes = self._compute_routes(slotting_plan, batches)
        objectives = self._evaluate_objectives(slotting_plan, batches, routes)
        
        solution = ParetoSolution(
            slotting_plan=slotting_plan,
            batches=batches,
            routes=routes,
            objectives=objectives
        )
        
        history = [{
            'generation': 0,
            'best_travel_distance': objectives[ObjectiveType.TRAVEL_DISTANCE.value],
            'best_workload_balance': objectives[ObjectiveType.WORKLOAD_BALANCE.value],
            'pareto_front_size': 1
        }]
        
        print(f"  Complete. Distance={objectives[ObjectiveType.TRAVEL_DISTANCE.value]:.1f}")
        
        return [solution], history


class RandomBaseline(BaselineAlgorithm):
    """Random assignment baseline."""
    
    def run(self) -> Tuple[List[ParetoSolution], List[Dict[str, Any]]]:
        """Run random baseline."""
        print("Running Random Baseline...")
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        solutions = []
        
        # Generate multiple random solutions
        for _ in range(min(50, self.config.population_size)):
            skus = list(self.warehouse.skus)
            locs = list(self.storage_locations)
            random.shuffle(skus)
            random.shuffle(locs)
            
            slotting_plan = {}
            for i, sku in enumerate(skus):
                if i < len(locs):
                    slotting_plan[sku.sku_id] = locs[i].loc_id
            
            orders = list(self.warehouse.orders)
            random.shuffle(orders)
            batches = []
            for i in range(0, len(orders), self.config.max_batch_size):
                batches.append([o.order_id for o in orders[i:i+self.config.max_batch_size]])
            
            routes = self._compute_routes(slotting_plan, batches)
            objectives = self._evaluate_objectives(slotting_plan, batches, routes)
            
            solutions.append(ParetoSolution(
                slotting_plan=slotting_plan,
                batches=batches,
                routes=routes,
                objectives=objectives
            ))
        
        pareto_front = self._get_non_dominated(solutions)
        
        history = [{
            'generation': 0,
            'best_travel_distance': min(s.objectives[ObjectiveType.TRAVEL_DISTANCE.value] 
                                        for s in pareto_front),
            'best_workload_balance': min(s.objectives[ObjectiveType.WORKLOAD_BALANCE.value] 
                                         for s in pareto_front),
            'pareto_front_size': len(pareto_front)
        }]
        
        print(f"  Complete. Pareto front: {len(pareto_front)} solutions")
        
        return pareto_front, history
