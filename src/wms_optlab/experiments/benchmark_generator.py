"""
Benchmark Instance Generator for Warehouse Optimization Experiments.

Generates standardized benchmark instances following patterns from the literature:
- Small (S): 50-100 SKUs, 30-50 locations, 200-500 orders
- Medium (M): 200-500 SKUs, 100-200 locations, 1000-3000 orders
- Large (L): 1000-2000 SKUs, 500-1000 locations, 10000+ orders

Instance characteristics:
- SKU demand follows Pareto/power-law distribution (80-20 rule)
- Location layouts: single-aisle, parallel-aisle, fishbone
- Order profiles: uniform, skewed, seasonal
"""

import random
import string
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from ..data.models import SKU, Location, Order, OrderLine, Warehouse


class InstanceSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"
    HUGE = "huge"       # Added for extended benchmarks
    MASSIVE = "massive" # Added for extended benchmarks


class LayoutType(Enum):
    PARALLEL_AISLE = "parallel_aisle"
    SINGLE_AISLE = "single_aisle"
    FISHBONE = "fishbone"
    FLYING_V = "flying_v"


class DemandProfile(Enum):
    PARETO = "pareto"  # 80-20 rule
    UNIFORM = "uniform"
    SEASONAL = "seasonal"
    CLUSTERED = "clustered"


@dataclass
class InstanceConfig:
    """Configuration for generating a benchmark instance."""
    size: InstanceSize = InstanceSize.MEDIUM
    layout_type: LayoutType = LayoutType.PARALLEL_AISLE
    demand_profile: DemandProfile = DemandProfile.PARETO
    
    # Size parameters (overridden by size preset if not specified)
    num_skus: Optional[int] = None
    num_locations: Optional[int] = None
    num_orders: Optional[int] = None
    
    # Layout parameters
    num_aisles: int = 10
    slots_per_aisle: int = 20
    aisle_width: float = 5.0
    slot_depth: float = 2.0
    
    # Order parameters
    lines_per_order: Tuple[int, int] = (1, 8)
    qty_per_line: Tuple[int, int] = (1, 10)
    
    # SKU parameters
    weight_range: Tuple[float, float] = (0.1, 50.0)
    volume_range: Tuple[float, float] = (0.01, 0.5)
    
    # Demand parameters
    pareto_alpha: float = 0.8  # Power law exponent
    
    random_seed: Optional[int] = 42


@dataclass
class BenchmarkInstance:
    """A generated benchmark instance."""
    warehouse: Warehouse
    config: InstanceConfig
    instance_id: str
    metadata: Dict
    
    # Ground truth for validation
    optimal_slotting: Optional[Dict[str, str]] = None
    
    def summary(self) -> str:
        """Return a summary string of the instance."""
        return (
            f"Instance {self.instance_id}: "
            f"{len(self.warehouse.skus)} SKUs, "
            f"{len(self.warehouse.locations)} locations, "
            f"{len(self.warehouse.orders)} orders"
        )


class BenchmarkInstanceGenerator:
    """Generates benchmark instances for warehouse optimization experiments."""
    
    # Note: locations includes depot, so storage = locations - 1
    # Ensure num_locations > num_skus for valid slotting
    # Updated with larger instances per reviewer feedback
    SIZE_PRESETS = {
        InstanceSize.SMALL: {'skus': 100, 'locations': 150, 'orders': 500},
        InstanceSize.MEDIUM: {'skus': 400, 'locations': 600, 'orders': 2000},
        InstanceSize.LARGE: {'skus': 1000, 'locations': 1500, 'orders': 8000},
        InstanceSize.XLARGE: {'skus': 2000, 'locations': 3000, 'orders': 20000},
        # Extended sizes to address reviewer concern about benchmark coverage
        InstanceSize.HUGE: {'skus': 3000, 'locations': 3600, 'orders': 30000},
        InstanceSize.MASSIVE: {'skus': 5000, 'locations': 6000, 'orders': 50000}
    }
    
    def __init__(self, config: Optional[InstanceConfig] = None):
        self.config = config or InstanceConfig()
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
    
    def generate(self, instance_id: Optional[str] = None) -> BenchmarkInstance:
        """Generate a complete benchmark instance."""
        if instance_id is None:
            instance_id = f"INST-{random.randint(1000, 9999)}"
        
        # Get size parameters
        preset = self.SIZE_PRESETS[self.config.size]
        num_skus = self.config.num_skus or preset['skus']
        num_locations = self.config.num_locations or preset['locations']
        num_orders = self.config.num_orders or preset['orders']
        
        # Generate components
        skus = self._generate_skus(num_skus)
        locations = self._generate_locations(num_locations)
        orders = self._generate_orders(skus, num_orders)
        
        warehouse = Warehouse(skus=skus, locations=locations, orders=orders)
        
        metadata = {
            'size': self.config.size.value,
            'layout_type': self.config.layout_type.value,
            'demand_profile': self.config.demand_profile.value,
            'num_skus': num_skus,
            'num_locations': num_locations,
            'num_orders': num_orders,
            'total_order_lines': sum(len(o.order_lines) for o in orders),
            'avg_lines_per_order': np.mean([len(o.order_lines) for o in orders]),
            'sku_demand_gini': self._calculate_gini_coefficient(skus, orders)
        }
        
        return BenchmarkInstance(
            warehouse=warehouse,
            config=self.config,
            instance_id=instance_id,
            metadata=metadata
        )
    
    def _generate_skus(self, num_skus: int) -> List[SKU]:
        """Generate SKU master data."""
        skus = []
        
        for i in range(num_skus):
            sku_id = f"SKU-{i:05d}"
            name = f"Product-{i:05d}"
            
            # ABC classification embedded in weight/volume distribution
            abc_class = 'A' if i < num_skus * 0.2 else ('B' if i < num_skus * 0.5 else 'C')
            
            # Fast movers tend to be smaller
            if abc_class == 'A':
                weight = random.uniform(0.1, 10.0)
                volume = random.uniform(0.01, 0.1)
            elif abc_class == 'B':
                weight = random.uniform(1.0, 25.0)
                volume = random.uniform(0.05, 0.25)
            else:
                weight = random.uniform(5.0, 50.0)
                volume = random.uniform(0.1, 0.5)
            
            skus.append(SKU(
                sku_id=sku_id,
                name=name,
                description=f"Class {abc_class} item",
                weight=round(weight, 2),
                volume=round(volume, 3)
            ))
        
        return skus
    
    def _generate_locations(self, num_locations: int) -> List[Location]:
        """Generate warehouse locations based on layout type."""
        locations = []
        
        # Always add depot first
        locations.append(Location(
            loc_id="DEPOT",
            coordinates=(0.0, 0.0, 0.0),
            capacity=10000.0,
            location_type="depot"
        ))
        
        if self.config.layout_type == LayoutType.PARALLEL_AISLE:
            locations.extend(self._generate_parallel_aisle_layout(num_locations - 1))
        elif self.config.layout_type == LayoutType.FISHBONE:
            locations.extend(self._generate_fishbone_layout(num_locations - 1))
        elif self.config.layout_type == LayoutType.FLYING_V:
            locations.extend(self._generate_flying_v_layout(num_locations - 1))
        else:
            locations.extend(self._generate_single_aisle_layout(num_locations - 1))
        
        return locations
    
    def _generate_parallel_aisle_layout(self, num_locations: int) -> List[Location]:
        """Generate standard parallel aisle warehouse layout."""
        locations = []
        
        # Calculate aisles and slots needed
        slots_needed = num_locations
        num_aisles = self.config.num_aisles
        slots_per_aisle = math.ceil(slots_needed / (num_aisles * 2))  # Both sides
        
        loc_count = 0
        for aisle in range(num_aisles):
            aisle_x = (aisle + 1) * self.config.aisle_width
            
            # Left side of aisle
            for slot in range(slots_per_aisle):
                if loc_count >= num_locations:
                    break
                loc_id = f"A{aisle+1:02d}-L{slot+1:02d}"
                y = (slot + 1) * self.config.slot_depth
                locations.append(Location(
                    loc_id=loc_id,
                    coordinates=(aisle_x - 0.5, y, 0.0),
                    capacity=10.0,
                    location_type="storage"
                ))
                loc_count += 1
            
            # Right side of aisle
            for slot in range(slots_per_aisle):
                if loc_count >= num_locations:
                    break
                loc_id = f"A{aisle+1:02d}-R{slot+1:02d}"
                y = (slot + 1) * self.config.slot_depth
                locations.append(Location(
                    loc_id=loc_id,
                    coordinates=(aisle_x + 0.5, y, 0.0),
                    capacity=10.0,
                    location_type="storage"
                ))
                loc_count += 1
        
        return locations
    
    def _generate_fishbone_layout(self, num_locations: int) -> List[Location]:
        """Generate fishbone warehouse layout."""
        locations = []
        
        # Calculate aisles and slots needed
        # Fishbone: 2 sides × (num_aisles/2) aisles × slots_per_aisle
        # So total = num_aisles × slots_per_aisle
        num_aisles = max(self.config.num_aisles, 10)  # Ensure minimum aisles
        slots_per_aisle = math.ceil(num_locations / num_aisles) + 1
        
        # Fishbone has a central cross-aisle with diagonal access
        center_y = 50.0
        loc_count = 0
        
        # Main aisles on each side of center
        for side in [1, -1]:
            for aisle in range(num_aisles // 2):
                aisle_x = (aisle + 1) * self.config.aisle_width * side + 25.0
                
                for slot in range(slots_per_aisle):
                    if loc_count >= num_locations:
                        break
                    
                    side_label = "E" if side > 0 else "W"
                    loc_id = f"FB{side_label}{aisle+1:02d}-{slot+1:02d}"
                    
                    # Angled slots
                    angle = 0.3 if side > 0 else -0.3
                    y = center_y + (slot + 1) * self.config.slot_depth
                    x = aisle_x + slot * angle
                    
                    locations.append(Location(
                        loc_id=loc_id,
                        coordinates=(x, y, 0.0),
                        capacity=10.0,
                        location_type="storage"
                    ))
                    loc_count += 1
        
        return locations
    
    def _generate_flying_v_layout(self, num_locations: int) -> List[Location]:
        """Generate Flying-V warehouse layout."""
        locations = []
        
        # Flying-V has V-shaped cross aisles
        loc_count = 0
        
        for aisle in range(self.config.num_aisles):
            aisle_x = (aisle + 1) * self.config.aisle_width
            
            for slot in range(self.config.slots_per_aisle):
                if loc_count >= num_locations:
                    break
                
                loc_id = f"FV{aisle+1:02d}-{slot+1:02d}"
                y = (slot + 1) * self.config.slot_depth
                
                # V-shape adjustment
                v_offset = abs(slot - self.config.slots_per_aisle / 2) * 0.2
                
                locations.append(Location(
                    loc_id=loc_id,
                    coordinates=(aisle_x + v_offset, y, 0.0),
                    capacity=10.0,
                    location_type="storage"
                ))
                loc_count += 1
        
        return locations
    
    def _generate_single_aisle_layout(self, num_locations: int) -> List[Location]:
        """Generate single aisle (linear) warehouse layout."""
        locations = []
        
        for i in range(num_locations):
            loc_id = f"S{i+1:04d}"
            y = (i + 1) * self.config.slot_depth
            side = -0.5 if i % 2 == 0 else 0.5
            
            locations.append(Location(
                loc_id=loc_id,
                coordinates=(side, y, 0.0),
                capacity=10.0,
                location_type="storage"
            ))
        
        return locations
    
    def _generate_orders(self, skus: List[SKU], num_orders: int) -> List[Order]:
        """Generate orders with specified demand profile."""
        orders = []
        
        # Generate SKU popularity distribution
        sku_popularity = self._generate_demand_distribution(len(skus))
        
        for i in range(num_orders):
            order_id = f"ORD-{i:06d}"
            
            # Number of lines in this order
            num_lines = random.randint(*self.config.lines_per_order)
            num_lines = min(num_lines, len(skus))
            
            # Select SKUs based on popularity
            selected_indices = np.random.choice(
                len(skus),
                size=num_lines,
                replace=False,
                p=sku_popularity
            )
            
            order_lines = []
            for idx in selected_indices:
                qty = random.randint(*self.config.qty_per_line)
                order_lines.append(OrderLine(sku=skus[idx], quantity=qty))
            
            # Generate due date (spread over 5 days)
            day = random.randint(1, 5)
            hour = random.randint(8, 17)
            due_date = f"2025-12-{day:02d}T{hour:02d}:00:00Z"
            
            orders.append(Order(
                order_id=order_id,
                order_lines=order_lines,
                due_date=due_date
            ))
        
        return orders
    
    def _generate_demand_distribution(self, num_skus: int) -> np.ndarray:
        """Generate demand probability distribution based on profile."""
        if self.config.demand_profile == DemandProfile.UNIFORM:
            return np.ones(num_skus) / num_skus
        
        elif self.config.demand_profile == DemandProfile.PARETO:
            # Power law distribution (80-20 rule)
            ranks = np.arange(1, num_skus + 1)
            probs = 1.0 / (ranks ** self.config.pareto_alpha)
            return probs / probs.sum()
        
        elif self.config.demand_profile == DemandProfile.SEASONAL:
            # Mix of high-demand seasonal items and stable items
            probs = np.zeros(num_skus)
            # 20% seasonal items with very high demand
            seasonal_count = int(num_skus * 0.2)
            probs[:seasonal_count] = 5.0
            # Rest follows mild Pareto
            probs[seasonal_count:] = 1.0 / (np.arange(seasonal_count + 1, num_skus + 1) ** 0.5)
            return probs / probs.sum()
        
        elif self.config.demand_profile == DemandProfile.CLUSTERED:
            # Multiple clusters of popular items
            probs = np.zeros(num_skus)
            num_clusters = 5
            cluster_size = num_skus // num_clusters
            for c in range(num_clusters):
                start = c * cluster_size
                end = min(start + cluster_size // 3, num_skus)  # Top third of each cluster
                probs[start:end] = 3.0 + random.random()
            probs[probs == 0] = 0.5
            return probs / probs.sum()
        
        return np.ones(num_skus) / num_skus
    
    def _calculate_gini_coefficient(self, skus: List[SKU], orders: List[Order]) -> float:
        """Calculate Gini coefficient of SKU demand distribution."""
        from collections import Counter
        
        demand = Counter()
        for order in orders:
            for line in order.order_lines:
                demand[line.sku.sku_id] += line.quantity
        
        values = sorted([demand.get(sku.sku_id, 0) for sku in skus])
        n = len(values)
        if n == 0 or sum(values) == 0:
            return 0.0
        
        cumsum = np.cumsum(values)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return round(gini, 4)
    
    @classmethod
    def generate_benchmark_suite(cls, 
                                  sizes: List[InstanceSize] = None,
                                  layouts: List[LayoutType] = None,
                                  profiles: List[DemandProfile] = None,
                                  instances_per_config: int = 5) -> List[BenchmarkInstance]:
        """
        Generate a complete benchmark suite for rigorous experimental evaluation.
        
        Default configuration generates 48 instances:
        - 4 sizes (S, M, L, XL) × 3 layouts (PAR, FIS, FLY) × 2 profiles (PAR, UNI) × 2 reps
        
        This addresses reviewer concerns about limited benchmark coverage.
        """
        if sizes is None:
            sizes = [InstanceSize.SMALL, InstanceSize.MEDIUM, InstanceSize.LARGE, InstanceSize.XLARGE]
        if layouts is None:
            layouts = [LayoutType.PARALLEL_AISLE, LayoutType.FISHBONE, LayoutType.FLYING_V]
        if profiles is None:
            profiles = [DemandProfile.PARETO, DemandProfile.UNIFORM]
        
        instances = []
        instance_num = 0
        
        for size in sizes:
            for layout in layouts:
                for profile in profiles:
                    for rep in range(instances_per_config):
                        config = InstanceConfig(
                            size=size,
                            layout_type=layout,
                            demand_profile=profile,
                            random_seed=42 + instance_num
                        )
                        generator = cls(config)
                        instance_id = f"{size.value[0].upper()}-{layout.value[:3].upper()}-{profile.value[:3].upper()}-{rep+1:02d}"
                        instance = generator.generate(instance_id)
                        instances.append(instance)
                        instance_num += 1
        
        return instances
    
    @classmethod
    def generate_eswa_revision_suite(cls) -> List[BenchmarkInstance]:
        """
        Generate the expanded benchmark suite for ESWA revision.
        
        Addresses reviewer concerns:
        - 12 distinct configurations (was 4)
        - Includes Large and XLarge instances (up to 2000 SKUs, 20000 orders)
        - Tests all 3 layout types (parallel, fishbone, flying-v)
        - Both Pareto and Uniform demand profiles
        - 5 replications per configuration for statistical power
        
        Total: 60 instances (12 configs × 5 reps)
        """
        configs = [
            # Small instances - fast tests
            (InstanceSize.SMALL, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
            (InstanceSize.SMALL, LayoutType.FISHBONE, DemandProfile.PARETO),
            (InstanceSize.SMALL, LayoutType.FLYING_V, DemandProfile.PARETO),
            
            # Medium instances - main results
            (InstanceSize.MEDIUM, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
            (InstanceSize.MEDIUM, LayoutType.FISHBONE, DemandProfile.PARETO),
            (InstanceSize.MEDIUM, LayoutType.FLYING_V, DemandProfile.PARETO),
            (InstanceSize.MEDIUM, LayoutType.PARALLEL_AISLE, DemandProfile.UNIFORM),
            (InstanceSize.MEDIUM, LayoutType.FISHBONE, DemandProfile.UNIFORM),
            
            # Large instances - scalability
            (InstanceSize.LARGE, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
            (InstanceSize.LARGE, LayoutType.FISHBONE, DemandProfile.PARETO),
            
            # XLarge instances - stress test
            (InstanceSize.XLARGE, LayoutType.PARALLEL_AISLE, DemandProfile.PARETO),
            (InstanceSize.XLARGE, LayoutType.FISHBONE, DemandProfile.PARETO),
        ]
        
        instances = []
        for idx, (size, layout, profile) in enumerate(configs):
            for rep in range(5):  # 5 replications per config
                config = InstanceConfig(
                    size=size,
                    layout_type=layout,
                    demand_profile=profile,
                    random_seed=1000 + idx * 100 + rep
                )
                generator = cls(config)
                
                # Create descriptive instance ID
                size_code = {'small': 'S', 'medium': 'M', 'large': 'L', 'xlarge': 'XL'}[size.value]
                layout_code = {'parallel_aisle': 'PAR', 'fishbone': 'FIS', 'flying_v': 'FLV'}[layout.value]
                profile_code = {'pareto': 'P', 'uniform': 'U'}[profile.value]
                instance_id = f"{size_code}-{layout_code}-{profile_code}-R{rep+1}"
                
                instance = generator.generate(instance_id)
                instances.append(instance)
        
        return instances
