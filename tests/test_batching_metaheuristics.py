import pytest
from wms_optlab.data.models import SKU, Order, OrderLine
from wms_optlab.batching.metaheuristics import simulated_annealing_batching

# Dummy cost function for testing: just sum the number of lines.
# A real cost function would calculate travel distance.
def simple_cost_function(batch: list):
    return sum(len(order.order_lines) for order in batch)

@pytest.fixture
def sample_orders():
    """Provides sample orders for testing batching."""
    skus = [SKU(sku_id=f"S{i}", name="") for i in range(5)]
    orders = [
        Order(order_id="O1", order_lines=[OrderLine(skus[0], 1)]),
        Order(order_id="O2", order_lines=[OrderLine(skus[1], 1), OrderLine(skus[2], 1)]),
        Order(order_id="O3", order_lines=[OrderLine(skus[3], 1)]),
        Order(order_id="O4", order_lines=[OrderLine(skus[4], 1)]),
        Order(order_id="O5", order_lines=[OrderLine(skus[0], 1), OrderLine(skus[4], 1)]),
    ]
    return orders

def test_simulated_annealing_batching_structure(sample_orders):
    """
    Tests that the simulated annealing batching function returns a valid batch structure.
    """
    batches = simulated_annealing_batching(
        sample_orders,
        cost_function=simple_cost_function,
        initial_temp=10,
        max_iterations=100
    )
    
    # All orders should be present in the final batches
    num_orders_in_batches = sum(len(b) for b in batches)
    assert num_orders_in_batches == len(sample_orders)
    
    # Check that the output is a list of lists (the batches)
    assert isinstance(batches, list)
    if batches:
        assert isinstance(batches[0], list)
        assert isinstance(batches[0][0], Order)


def test_simulated_annealing_handles_single_batch():
    """
    When only a single batch is possible, the function should not attempt
    to sample two batches and should return the initial grouping.
    """
    skus = [SKU(sku_id="S1", name="")]
    orders = [Order(order_id="O1", order_lines=[OrderLine(skus[0], 1)])]

    batches = simulated_annealing_batching(
        orders,
        cost_function=simple_cost_function,
        initial_temp=10,
        max_iterations=10
    )

    assert len(batches) == 1
    assert batches[0][0].order_id == "O1"
