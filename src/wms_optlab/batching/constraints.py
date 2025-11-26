from typing import List
from ..data.models import Order

def check_batch_weight_constraint(batch: List[Order], max_weight: float) -> bool:
    """Checks if a batch of orders satisfies a maximum weight constraint."""
    total_weight = 0.0
    for order in batch:
        for line in order.order_lines:
            total_weight += line.sku.weight * line.quantity
    return total_weight <= max_weight

def check_batch_volume_constraint(batch: List[Order], max_volume: float) -> bool:
    """Checks if a batch of orders satisfies a maximum volume constraint."""
    total_volume = 0.0
    for order in batch:
        for line in order.order_lines:
            total_volume += line.sku.volume * line.quantity
    return total_volume <= max_volume
