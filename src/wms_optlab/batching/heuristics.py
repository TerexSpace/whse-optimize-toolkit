from typing import List
from ..data.models import Order

def batch_by_due_date(orders: List[Order], batch_size: int) -> List[List[Order]]:
    """
    Groups orders into batches based on their due date.
    Orders with earlier due dates are prioritized.
    """
    # Sort orders by due date (assuming ISO format string)
    sorted_orders = sorted(orders, key=lambda o: o.due_date)
    
    # Create batches of a fixed size
    batches = [sorted_orders[i:i + batch_size] for i in range(0, len(sorted_orders), batch_size)]
    return batches
