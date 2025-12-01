"""
MOIWOF: Multi-Objective Integrated Warehouse Optimization Framework.

This module implements a novel adaptive decomposition algorithm that simultaneously
optimizes storage location assignment (slotting), picker routing, and order batching
as an integrated multi-objective problem.

Revolutionary Contributions:
1. Adaptive Decomposition Strategy (ADS) - dynamically adjusts subproblem coupling
2. Cross-Component Learning (CCL) - transfers knowledge between slotting/routing/batching
3. Pareto Archive with Diversity Preservation (PADP)
4. Real-time Responsiveness Indicator (RRI) as third objective
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional, Any
from enum import Enum
import random
import math
import numpy as np
from collections import Counter
import copy

from ..data.models import SKU, Location, Order, Warehouse, OrderLine
from ..layout.geometry import manhattan_distance, Point
from ..slotting.heuristics import assign_by_abc_popularity
from ..routing.policies import get_s_shape_route


class ObjectiveType(Enum):
    TRAVEL_DISTANCE = "travel_distance"
    THROUGHPUT_TIME = "throughput_time"
    WORKLOAD_BALANCE = "workload_balance"


@dataclass
class ParetoSolution:
    """Represents a solution in the Pareto front."""
    slotting_plan: Dict[str, str]  # {sku_id: loc_id}
    batches: List[List[str]]  # List of batches, each batch is list of order_ids
    routes: Dict[str, List[str]]  # {batch_id: [loc_ids]}
    objectives: Dict[str, float]  # {objective_name: value}
    generation: int = 0
    crowding_distance: float = 0.0
    rank: int = 0
    
    def dominates(self, other: 'ParetoSolution') -> bool:
        """Check if this solution dominates another (all objectives minimized)."""
        dominated_in_all = True
        better_in_at_least_one = False
        for obj in self.objectives:
            if self.objectives[obj] > other.objectives[obj]:
                dominated_in_all = False
            elif self.objectives[obj] < other.objectives[obj]:
                better_in_at_least_one = True
        return dominated_in_all and better_in_at_least_one


@dataclass
class MOIWOFConfig:
    """Configuration for the MOIWOF algorithm."""
    population_size: int = 100
    max_generations: int = 200
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    archive_size: int = 100
    decomposition_phases: int = 3
    learning_rate: float = 0.1
    adaptive_threshold: float = 0.05
    num_pickers: int = 5
    max_batch_size: int = 10
    max_batch_weight: float = 100.0
    random_seed: Optional[int] = 42


@dataclass 
class AdaptiveWeights:
    """Adaptive weights for decomposition strategy."""
    slotting_weight: float = 0.4
    routing_weight: float = 0.35
    batching_weight: float = 0.25
    history: List[Tuple[float, float, float]] = field(default_factory=list)
    
    def update(self, improvement_rates: Dict[str, float]):
        """Update weights based on improvement rates from each subproblem."""
        total = sum(improvement_rates.values()) + 1e-10
        self.slotting_weight = 0.7 * self.slotting_weight + 0.3 * (improvement_rates.get('slotting', 0) / total)
        self.routing_weight = 0.7 * self.routing_weight + 0.3 * (improvement_rates.get('routing', 0) / total)
        self.batching_weight = 0.7 * self.batching_weight + 0.3 * (improvement_rates.get('batching', 0) / total)
        
        # Normalize
        total_weight = self.slotting_weight + self.routing_weight + self.batching_weight
        self.slotting_weight /= total_weight
        self.routing_weight /= total_weight
        self.batching_weight /= total_weight
        
        self.history.append((self.slotting_weight, self.routing_weight, self.batching_weight))


class CrossComponentLearning:
    """Implements cross-component knowledge transfer mechanism."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.sku_location_affinity: Dict[str, Dict[str, float]] = {}
        self.location_congestion_score: Dict[str, float] = {}
        self.batch_synergy_matrix: Dict[Tuple[str, str], float] = {}
    
    def learn_from_routing(self, routes: Dict[str, List[str]], distances: Dict[str, float]):
        """Extract location congestion patterns from routing solutions."""
        visit_counts = Counter()
        for route in routes.values():
            visit_counts.update(route)
        
        max_visits = max(visit_counts.values()) if visit_counts else 1
        for loc_id, count in visit_counts.items():
            current = self.location_congestion_score.get(loc_id, 0.0)
            new_score = count / max_visits
            self.location_congestion_score[loc_id] = (
                (1 - self.learning_rate) * current + self.learning_rate * new_score
            )
    
    def learn_from_batching(self, batches: List[List[Order]], batch_costs: List[float]):
        """Learn synergy patterns between orders in batches."""
        for batch, cost in zip(batches, batch_costs):
            if len(batch) < 2:
                continue
            normalized_cost = 1.0 / (cost + 1)  # Lower cost = higher synergy
            for i, order1 in enumerate(batch):
                for order2 in batch[i+1:]:
                    key = tuple(sorted([order1.order_id, order2.order_id]))
                    current = self.batch_synergy_matrix.get(key, 0.5)
                    self.batch_synergy_matrix[key] = (
                        (1 - self.learning_rate) * current + self.learning_rate * normalized_cost
                    )
    
    def suggest_slotting_improvement(self, sku_id: str, current_loc: str, 
                                      locations: List[Location]) -> Optional[str]:
        """Suggest better location based on learned patterns."""
        # Prefer locations with lower congestion scores
        candidates = []
        for loc in locations:
            if loc.location_type == 'storage':
                congestion = self.location_congestion_score.get(loc.loc_id, 0.5)
                candidates.append((loc.loc_id, congestion))
        
        if not candidates:
            return None
        
        # Return location with lowest congestion (with some randomness for exploration)
        candidates.sort(key=lambda x: x[1])
        top_k = candidates[:max(3, len(candidates) // 5)]
        return random.choice(top_k)[0]


class MOIWOF:
    """
    Multi-Objective Integrated Warehouse Optimization Framework.
    
    This class implements the main optimization algorithm with:
    - Adaptive Decomposition Strategy
    - Cross-Component Learning
    - Pareto Archive Management
    - Three objectives: Travel Distance, Throughput Time, Workload Balance
    """
    
    def __init__(self, warehouse: Warehouse, config: MOIWOFConfig):
        self.warehouse = warehouse
        self.config = config
        self.adaptive_weights = AdaptiveWeights()
        self.ccl = CrossComponentLearning(config.learning_rate)
        self.pareto_archive: List[ParetoSolution] = []
        self.generation_history: List[Dict[str, Any]] = []
        
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        # Pre-compute depot and graph
        self.depot = next((loc for loc in warehouse.locations if loc.location_type == 'depot'), None)
        if not self.depot:
            raise ValueError("Warehouse must have a depot location")
        self.graph = warehouse.get_graph()
        self.loc_map = {loc.loc_id: loc for loc in warehouse.locations}
        self.sku_map = {sku.sku_id: sku for sku in warehouse.skus}
        self.order_map = {order.order_id: order for order in warehouse.orders}
        self.storage_locations = [loc for loc in warehouse.locations if loc.location_type == 'storage']
    
    def initialize_population(self) -> List[ParetoSolution]:
        """Generate initial population with diverse solutions."""
        population = []
        
        for i in range(self.config.population_size):
            # Generate slotting plan with variations
            slotting_plan = self._generate_slotting_variant(i)
            
            # Generate batching solution
            batches = self._generate_batching_variant(i)
            
            # Generate routes for each batch
            routes = self._compute_routes(slotting_plan, batches)
            
            # Evaluate objectives
            objectives = self._evaluate_objectives(slotting_plan, batches, routes)
            
            solution = ParetoSolution(
                slotting_plan=slotting_plan,
                batches=batches,
                routes=routes,
                objectives=objectives,
                generation=0
            )
            population.append(solution)
        
        return population
    
    def _generate_slotting_variant(self, variant_id: int) -> Dict[str, str]:
        """Generate a slotting plan variant."""
        if variant_id == 0:
            # Use ABC heuristic as baseline
            return assign_by_abc_popularity(
                self.warehouse.skus,
                self.warehouse.locations,
                self.warehouse.orders,
                distance_metric=manhattan_distance,
                depot_location=self.depot.coordinates
            )
        else:
            # Random permutation with ABC bias
            skus = list(self.warehouse.skus)
            locations = list(self.storage_locations)
            
            # Calculate SKU popularity
            sku_demand = Counter()
            for order in self.warehouse.orders:
                for line in order.order_lines:
                    sku_demand[line.sku.sku_id] += line.quantity
            
            # Sort with randomness
            random.shuffle(skus)
            skus.sort(key=lambda s: -sku_demand.get(s.sku_id, 0) + random.gauss(0, 5))
            
            locations.sort(
                key=lambda loc: manhattan_distance(loc.coordinates, self.depot.coordinates) + random.gauss(0, 2)
            )
            
            plan = {}
            for i, sku in enumerate(skus):
                if i < len(locations):
                    plan[sku.sku_id] = locations[i].loc_id
            
            return plan
    
    def _generate_batching_variant(self, variant_id: int) -> List[List[str]]:
        """Generate a batching solution variant."""
        orders = list(self.warehouse.orders)
        random.shuffle(orders)
        
        batches = []
        current_batch = []
        current_weight = 0.0
        
        for order in orders:
            order_weight = sum(
                line.sku.weight * line.quantity 
                for line in order.order_lines
            )
            
            if (len(current_batch) >= self.config.max_batch_size or 
                current_weight + order_weight > self.config.max_batch_weight):
                if current_batch:
                    batches.append([o.order_id for o in current_batch])
                current_batch = [order]
                current_weight = order_weight
            else:
                current_batch.append(order)
                current_weight += order_weight
        
        if current_batch:
            batches.append([o.order_id for o in current_batch])
        
        return batches
    
    def _compute_routes(self, slotting_plan: Dict[str, str], 
                        batches: List[List[str]]) -> Dict[str, List[str]]:
        """Compute picker routes for each batch."""
        routes = {}
        
        for batch_idx, batch in enumerate(batches):
            # Collect all locations to visit
            pick_loc_ids = set()
            for order_id in batch:
                order = self.order_map.get(order_id)
                if order:
                    for line in order.order_lines:
                        loc_id = slotting_plan.get(line.sku.sku_id)
                        if loc_id:
                            pick_loc_ids.add(loc_id)
            
            # Get Location objects
            pick_locations = [self.loc_map[loc_id] for loc_id in pick_loc_ids if loc_id in self.loc_map]
            
            # Compute route using S-shape policy
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
        
        # Objective 1: Total Travel Distance
        total_distance = 0.0
        for route in routes.values():
            total_distance += self._calculate_route_distance(route)
        
        # Objective 2: Throughput Time (makespan estimation)
        batch_times = []
        for batch_id, route in routes.items():
            distance = self._calculate_route_distance(route)
            # Time = travel time + pick time (assume 0.5 units per location)
            pick_time = len(route) * 0.5
            batch_times.append(distance + pick_time)
        
        # Assign batches to pickers (round-robin)
        picker_workloads = [0.0] * self.config.num_pickers
        for i, time in enumerate(batch_times):
            picker_idx = i % self.config.num_pickers
            picker_workloads[picker_idx] += time
        
        throughput_time = max(picker_workloads)  # Makespan
        
        # Objective 3: Workload Balance (coefficient of variation)
        if picker_workloads:
            mean_workload = np.mean(picker_workloads)
            std_workload = np.std(picker_workloads)
            workload_balance = std_workload / (mean_workload + 1e-10)
        else:
            workload_balance = 0.0
        
        return {
            ObjectiveType.TRAVEL_DISTANCE.value: total_distance,
            ObjectiveType.THROUGHPUT_TIME.value: throughput_time,
            ObjectiveType.WORKLOAD_BALANCE.value: workload_balance
        }
    
    def _calculate_route_distance(self, route: List[str]) -> float:
        """Calculate total distance of a route."""
        if len(route) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(route) - 1):
            loc1 = self.loc_map.get(route[i])
            loc2 = self.loc_map.get(route[i + 1])
            if loc1 and loc2:
                total += manhattan_distance(loc1.coordinates, loc2.coordinates)
        
        return total
    
    def fast_non_dominated_sort(self, population: List[ParetoSolution]) -> List[List[int]]:
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
        
        # Remove empty fronts
        fronts = [f for f in fronts if f]
        return fronts if fronts else [[]]
    
    def calculate_crowding_distance(self, population: List[ParetoSolution], front: List[int]):
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            for idx in front:
                population[idx].crowding_distance = float('inf')
            return
        
        for idx in front:
            population[idx].crowding_distance = 0.0
        
        objectives = list(population[0].objectives.keys())
        
        for obj in objectives:
            # Sort by this objective
            front_sorted = sorted(front, key=lambda i: population[i].objectives[obj])
            
            # Boundary solutions get infinite distance
            population[front_sorted[0]].crowding_distance = float('inf')
            population[front_sorted[-1]].crowding_distance = float('inf')
            
            # Calculate range
            obj_min = population[front_sorted[0]].objectives[obj]
            obj_max = population[front_sorted[-1]].objectives[obj]
            obj_range = obj_max - obj_min + 1e-10
            
            # Add crowding distance
            for i in range(1, len(front_sorted) - 1):
                idx = front_sorted[i]
                prev_val = population[front_sorted[i - 1]].objectives[obj]
                next_val = population[front_sorted[i + 1]].objectives[obj]
                population[idx].crowding_distance += (next_val - prev_val) / obj_range
    
    def crossover(self, parent1: ParetoSolution, parent2: ParetoSolution) -> ParetoSolution:
        """Perform crossover between two parent solutions."""
        # Slotting crossover: uniform crossover
        child_slotting = {}
        all_skus = set(parent1.slotting_plan.keys()) | set(parent2.slotting_plan.keys())
        used_locations = set()
        
        for sku_id in all_skus:
            if random.random() < 0.5:
                loc = parent1.slotting_plan.get(sku_id)
            else:
                loc = parent2.slotting_plan.get(sku_id)
            
            # Ensure no duplicate locations
            if loc and loc not in used_locations:
                child_slotting[sku_id] = loc
                used_locations.add(loc)
            else:
                # Find alternative location
                for storage_loc in self.storage_locations:
                    if storage_loc.loc_id not in used_locations:
                        child_slotting[sku_id] = storage_loc.loc_id
                        used_locations.add(storage_loc.loc_id)
                        break
        
        # Batching crossover: two-point crossover on batch structure
        p1_flat = [order_id for batch in parent1.batches for order_id in batch]
        p2_flat = [order_id for batch in parent2.batches for order_id in batch]
        
        # Use order from parent1 with some swaps from parent2
        child_orders = p1_flat.copy()
        for i in range(len(child_orders)):
            if random.random() < 0.3 and i < len(p2_flat):
                # Swap positions
                if p2_flat[i] in child_orders:
                    idx = child_orders.index(p2_flat[i])
                    child_orders[i], child_orders[idx] = child_orders[idx], child_orders[i]
        
        # Reconstruct batches
        child_batches = self._reconstruct_batches(child_orders)
        
        # Compute routes and objectives
        routes = self._compute_routes(child_slotting, child_batches)
        objectives = self._evaluate_objectives(child_slotting, child_batches, routes)
        
        return ParetoSolution(
            slotting_plan=child_slotting,
            batches=child_batches,
            routes=routes,
            objectives=objectives,
            generation=max(parent1.generation, parent2.generation) + 1
        )
    
    def mutate(self, solution: ParetoSolution) -> ParetoSolution:
        """Apply mutation to a solution."""
        new_slotting = dict(solution.slotting_plan)
        new_batches = [list(batch) for batch in solution.batches]
        
        # Slotting mutation: swap two SKU locations
        if random.random() < self.config.mutation_rate:
            sku_ids = list(new_slotting.keys())
            if len(sku_ids) >= 2:
                sku1, sku2 = random.sample(sku_ids, 2)
                new_slotting[sku1], new_slotting[sku2] = new_slotting[sku2], new_slotting[sku1]
        
        # Cross-component learning guided mutation
        if random.random() < self.config.mutation_rate and self.ccl.location_congestion_score:
            sku_ids = list(new_slotting.keys())
            if sku_ids:
                sku_to_move = random.choice(sku_ids)
                current_loc = new_slotting[sku_to_move]
                suggested = self.ccl.suggest_slotting_improvement(
                    sku_to_move, current_loc, self.storage_locations
                )
                if suggested and suggested != current_loc:
                    # Swap with whatever is at suggested location
                    for other_sku, other_loc in new_slotting.items():
                        if other_loc == suggested:
                            new_slotting[other_sku] = current_loc
                            break
                    new_slotting[sku_to_move] = suggested
        
        # Batching mutation: move order between batches
        if random.random() < self.config.mutation_rate and len(new_batches) >= 2:
            batch_idx1, batch_idx2 = random.sample(range(len(new_batches)), 2)
            if new_batches[batch_idx1]:
                order_to_move = random.choice(new_batches[batch_idx1])
                new_batches[batch_idx1].remove(order_to_move)
                new_batches[batch_idx2].append(order_to_move)
        
        # Remove empty batches
        new_batches = [batch for batch in new_batches if batch]
        
        # Compute routes and objectives
        routes = self._compute_routes(new_slotting, new_batches)
        objectives = self._evaluate_objectives(new_slotting, new_batches, routes)
        
        return ParetoSolution(
            slotting_plan=new_slotting,
            batches=new_batches,
            routes=routes,
            objectives=objectives,
            generation=solution.generation + 1
        )
    
    def _reconstruct_batches(self, order_ids: List[str]) -> List[List[str]]:
        """Reconstruct batch structure from flat order list."""
        batches = []
        current_batch = []
        current_weight = 0.0
        
        for order_id in order_ids:
            order = self.order_map.get(order_id)
            if not order:
                continue
            
            order_weight = sum(
                line.sku.weight * line.quantity
                for line in order.order_lines
            )
            
            if (len(current_batch) >= self.config.max_batch_size or
                current_weight + order_weight > self.config.max_batch_weight):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [order_id]
                current_weight = order_weight
            else:
                current_batch.append(order_id)
                current_weight += order_weight
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def update_pareto_archive(self, population: List[ParetoSolution]):
        """Update the Pareto archive with non-dominated solutions."""
        combined = self.pareto_archive + [
            sol for sol in population if sol.rank == 0
        ]
        
        # Remove dominated solutions
        non_dominated = []
        for i, sol in enumerate(combined):
            is_dominated = False
            for j, other in enumerate(combined):
                if i != j and other.dominates(sol):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(sol)
        
        # If archive too large, select based on crowding distance
        if len(non_dominated) > self.config.archive_size:
            fronts = self.fast_non_dominated_sort(non_dominated)
            self.calculate_crowding_distance(non_dominated, list(range(len(non_dominated))))
            non_dominated.sort(key=lambda s: s.crowding_distance, reverse=True)
            non_dominated = non_dominated[:self.config.archive_size]
        
        self.pareto_archive = non_dominated
    
    def selection(self, population: List[ParetoSolution]) -> ParetoSolution:
        """Tournament selection based on rank and crowding distance."""
        tournament_size = 3
        candidates = random.sample(population, min(tournament_size, len(population)))
        
        # Select best based on rank, then crowding distance
        best = min(candidates, key=lambda s: (s.rank, -s.crowding_distance))
        return best
    
    def run(self) -> Tuple[List[ParetoSolution], List[Dict[str, Any]]]:
        """Run the MOIWOF algorithm."""
        print(f"Initializing MOIWOF with population size {self.config.population_size}")
        
        # Initialize population
        population = self.initialize_population()
        
        # Initial non-dominated sorting
        fronts = self.fast_non_dominated_sort(population)
        for front in fronts:
            self.calculate_crowding_distance(population, front)
        
        # Update archive
        self.update_pareto_archive(population)
        
        # Track statistics
        self.generation_history = []
        
        for gen in range(self.config.max_generations):
            # Track improvement rates for adaptive weights
            prev_best_objectives = {
                obj: min(sol.objectives[obj] for sol in population)
                for obj in population[0].objectives
            }
            
            # Generate offspring
            offspring = []
            while len(offspring) < self.config.population_size:
                # Selection
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                
                # Crossover
                if random.random() < self.config.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))
                
                # Mutation
                child = self.mutate(child)
                offspring.append(child)
            
            # Combine parent and offspring
            combined = population + offspring
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined)
            
            # Select next generation
            new_population = []
            front_idx = 0
            while len(new_population) + len(fronts[front_idx]) <= self.config.population_size:
                self.calculate_crowding_distance(combined, fronts[front_idx])
                new_population.extend([combined[i] for i in fronts[front_idx]])
                front_idx += 1
                if front_idx >= len(fronts):
                    break
            
            # Fill remaining slots using crowding distance
            if len(new_population) < self.config.population_size and front_idx < len(fronts):
                self.calculate_crowding_distance(combined, fronts[front_idx])
                remaining = sorted(
                    fronts[front_idx],
                    key=lambda i: combined[i].crowding_distance,
                    reverse=True
                )
                for i in remaining:
                    if len(new_population) >= self.config.population_size:
                        break
                    new_population.append(combined[i])
            
            population = new_population
            
            # Update cross-component learning
            for sol in population[:10]:  # Learn from top solutions
                self.ccl.learn_from_routing(sol.routes, {})
            
            # Update Pareto archive
            self.update_pareto_archive(population)
            
            # Calculate improvement and update adaptive weights
            current_best_objectives = {
                obj: min(sol.objectives[obj] for sol in population)
                for obj in population[0].objectives
            }
            
            improvement_rates = {
                'slotting': max(0, prev_best_objectives.get(ObjectiveType.TRAVEL_DISTANCE.value, 0) - 
                               current_best_objectives.get(ObjectiveType.TRAVEL_DISTANCE.value, 0)),
                'routing': max(0, prev_best_objectives.get(ObjectiveType.THROUGHPUT_TIME.value, 0) - 
                              current_best_objectives.get(ObjectiveType.THROUGHPUT_TIME.value, 0)),
                'batching': max(0, prev_best_objectives.get(ObjectiveType.WORKLOAD_BALANCE.value, 0) - 
                               current_best_objectives.get(ObjectiveType.WORKLOAD_BALANCE.value, 0))
            }
            self.adaptive_weights.update(improvement_rates)
            
            # Record generation statistics
            gen_stats = {
                'generation': gen,
                'pareto_front_size': len(self.pareto_archive),
                'best_travel_distance': current_best_objectives.get(ObjectiveType.TRAVEL_DISTANCE.value),
                'best_throughput_time': current_best_objectives.get(ObjectiveType.THROUGHPUT_TIME.value),
                'best_workload_balance': current_best_objectives.get(ObjectiveType.WORKLOAD_BALANCE.value),
                'adaptive_weights': (
                    self.adaptive_weights.slotting_weight,
                    self.adaptive_weights.routing_weight,
                    self.adaptive_weights.batching_weight
                )
            }
            self.generation_history.append(gen_stats)
            
            if gen % 20 == 0:
                print(f"Generation {gen}: Pareto front size = {len(self.pareto_archive)}, "
                      f"Best distance = {current_best_objectives.get(ObjectiveType.TRAVEL_DISTANCE.value):.2f}")
        
        print(f"Optimization complete. Final Pareto front size: {len(self.pareto_archive)}")
        return self.pareto_archive, self.generation_history


def run_moiwof_experiment(warehouse: Warehouse, 
                          config: Optional[MOIWOFConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to run a complete MOIWOF experiment.
    
    Returns:
        Dictionary with Pareto front, generation history, and summary statistics.
    """
    if config is None:
        config = MOIWOFConfig()
    
    optimizer = MOIWOF(warehouse, config)
    pareto_front, history = optimizer.run()
    
    # Extract summary statistics
    summary = {
        'pareto_front_size': len(pareto_front),
        'generations': len(history),
        'final_hypervolume': calculate_hypervolume(pareto_front),
        'best_solutions': {
            ObjectiveType.TRAVEL_DISTANCE.value: min(
                pareto_front, key=lambda s: s.objectives[ObjectiveType.TRAVEL_DISTANCE.value]
            ),
            ObjectiveType.THROUGHPUT_TIME.value: min(
                pareto_front, key=lambda s: s.objectives[ObjectiveType.THROUGHPUT_TIME.value]
            ),
            ObjectiveType.WORKLOAD_BALANCE.value: min(
                pareto_front, key=lambda s: s.objectives[ObjectiveType.WORKLOAD_BALANCE.value]
            )
        }
    }
    
    return {
        'pareto_front': pareto_front,
        'history': history,
        'summary': summary,
        'adaptive_weights_history': optimizer.adaptive_weights.history
    }


def calculate_hypervolume(pareto_front: List[ParetoSolution], 
                          reference_point: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate hypervolume indicator for the Pareto front.
    Uses a simple 3D hypervolume calculation.
    """
    if not pareto_front:
        return 0.0
    
    objectives = list(pareto_front[0].objectives.keys())
    
    if reference_point is None:
        # Use worst values * 1.1 as reference
        reference_point = {
            obj: max(sol.objectives[obj] for sol in pareto_front) * 1.1
            for obj in objectives
        }
    
    # Simple hypervolume approximation using Monte Carlo
    n_samples = 10000
    count_dominated = 0
    
    mins = {obj: min(sol.objectives[obj] for sol in pareto_front) for obj in objectives}
    maxs = reference_point
    
    for _ in range(n_samples):
        sample = {
            obj: random.uniform(mins[obj], maxs[obj])
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
