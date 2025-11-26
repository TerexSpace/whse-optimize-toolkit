import random
import string
import numpy as np
from typing import List, Tuple
from .models import SKU, Order, OrderLine

def generate_random_skus(num_skus: int) -> List[SKU]:
    """Generates a list of random SKUs."""
    skus = []
    for i in range(num_skus):
        sku_id = f"SKU-{i:04d}"
        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        weight = round(random.uniform(0.1, 50.0), 2)
        volume = round(random.uniform(0.01, 0.5), 3)
        skus.append(SKU(sku_id, name, weight=weight, volume=volume))
    return skus

def generate_synthetic_orders(
    skus: List[SKU],
    num_orders: int,
    lines_per_order: Tuple[int, int] = (1, 5),
    qty_per_line: Tuple[int, int] = (1, 10)
) -> List[Order]:
    """
    Generates a list of synthetic orders with a skewed (power law) demand distribution.
    """
    orders = []
    # Create a skewed distribution for SKU popularity (power law)
    popularity = np.random.power(a=0.8, size=len(skus))
    sku_popularity = popularity / popularity.sum()

    for i in range(num_orders):
        order_id = f"ORD-{i:05d}"
        num_lines = random.randint(*lines_per_order)
        order_lines = []

        # Sample SKUs based on the skewed popularity distribution
        chosen_skus = np.random.choice(skus, size=num_lines, p=sku_popularity, replace=False)

        for sku in chosen_skus:
            quantity = random.randint(*qty_per_line)
            order_lines.append(OrderLine(sku, quantity))

        orders.append(Order(order_id, order_lines))
    return orders
