from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class SKU:
    """Stock Keeping Unit."""
    sku_id: str
    name: str
    description: str = ""
    weight: float = 0.0  # in kg
    volume: float = 0.0  # in m^3

@dataclass(frozen=True)
class Location:
    """A storage location in the warehouse."""
    loc_id: str
    coordinates: Tuple[float, float, float]  # (x, y, z)
    capacity: float = 1.0  # e.g., in terms of volume or weight
    location_type: str = "storage"  # e.g., storage, P&D, depot

@dataclass
class OrderLine:
    """A single line item in a customer order."""
    sku: SKU
    quantity: int

@dataclass
class Order:
    """A customer order, consisting of multiple order lines."""
    order_id: str
    order_lines: List[OrderLine]
    due_date: str = "" # ISO 8601 format

@dataclass
class InventoryState:
    """Represents the inventory of SKUs at locations."""
    inventory: Dict[Location, SKU] = field(default_factory=dict)

    def add_item(self, sku: SKU, location: Location):
        self.inventory[location] = sku

    def get_sku_location(self, sku: SKU) -> Location | None:
        for loc, item_sku in self.inventory.items():
            if item_sku == sku:
                return loc
        return None

@dataclass
class Warehouse:
    """The main warehouse model, containing all its components."""
    skus: List[SKU]
    locations: List[Location]
    orders: List[Order]
    inventory_state: InventoryState = field(default_factory=InventoryState)
    _graph_cache: object = field(default=None, init=False, repr=False, compare=False)

    def get_graph(self, aisle_connectivity: List[Tuple[str, str]] | None = None):
        """
        Builds (and caches) a warehouse topology graph using Manhattan distances.

        Args:
            aisle_connectivity: Optional explicit connectivity between locations. If omitted,
                the graph is assumed fully connected.

        Returns:
            A networkx.Graph with edge weights representing travel distance.
        """
        from ..layout.topology import create_warehouse_graph

        # Cache only the default (fully connected) graph; custom connectivity
        # is recomputed to avoid mixing incompatible structures.
        if aisle_connectivity is None and self._graph_cache is not None:
            return self._graph_cache

        graph = create_warehouse_graph(self.locations, aisle_connectivity)
        if aisle_connectivity is None:
            self._graph_cache = graph
        return graph
