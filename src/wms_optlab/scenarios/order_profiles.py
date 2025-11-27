from typing import List, Dict
import pandas as pd
from ..data.models import Order

def analyze_order_profile(orders: List[Order]) -> Dict:
    """
    Analyzes a list of orders to produce a profile.
    This can include stats on lines per order, quantity per line, etc.
    """
    if not orders:
        return {}
        
    lines_per_order = [len(o.order_lines) for o in orders]
    qty_per_line = [line.quantity for o in orders for line in o.order_lines]
    
    profile = {
        "num_orders": len(orders),
        "avg_lines_per_order": sum(lines_per_order) / len(orders),
        "max_lines_per_order": max(lines_per_order),
        "avg_qty_per_line": sum(qty_per_line) / len(qty_per_line) if qty_per_line else 0,
    }
    return profile
