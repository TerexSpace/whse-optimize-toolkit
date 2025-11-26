import pytest
from wms_optlab.data.models import SKU, Location, Order, OrderLine
from wms_optlab.slotting.heuristics import assign_by_abc_popularity
from wms_optlab.layout.geometry import manhattan_distance

@pytest.fixture
def sample_data():
    """Provides sample data for testing slotting heuristics."""
    skus = [
        SKU(sku_id="S1", name="Popular"),
        SKU(sku_id="S2", name=" unpopular"),
    ]
    locations = [
        Location(loc_id="L1", coordinates=(10, 0, 0)), # Closer
        Location(loc_id="L2", coordinates=(20, 0, 0)), # Farther
    ]
    orders = [
        Order(order_id="O1", order_lines=[OrderLine(skus[0], 10)]),
        Order(order_id="O2", order_lines=[OrderLine(skus[0], 10)]),
        Order(order_id="O3", order_lines=[OrderLine(skus[1], 1)]),
    ]
    return skus, locations, orders

def test_assign_by_abc_popularity(sample_data):
    """
    Tests that the most popular SKU is assigned to the closest location.
    """
    skus, locations, orders = sample_data
    
    slotting_plan = assign_by_abc_popularity(
        skus,
        locations,
        orders,
        distance_metric=manhattan_distance,
        depot_location=(0, 0, 0)
    )

    # S1 is the most popular SKU, so it should be assigned to L1 (the closer location)
    assert slotting_plan.get("S1") == "L1"
    
    # S2 is less popular, so it should be assigned to L2
    assert slotting_plan.get("S2") == "L2"
    
    assert len(slotting_plan) == 2


def test_assign_includes_zero_demand_skus():
    """
    SKUs with zero observed demand should still be assigned to remaining locations.
    """
    skus = [
        SKU(sku_id="S1", name="Popular"),
        SKU(sku_id="S2", name="Less Popular"),
        SKU(sku_id="S3", name="New Item"),
    ]
    locations = [
        Location(loc_id="L1", coordinates=(10, 0, 0)),
        Location(loc_id="L2", coordinates=(20, 0, 0)),
        Location(loc_id="L3", coordinates=(30, 0, 0)),
    ]
    orders = [
        Order(order_id="O1", order_lines=[OrderLine(skus[0], 10)]),
        Order(order_id="O2", order_lines=[OrderLine(skus[1], 1)]),
    ]

    slotting_plan = assign_by_abc_popularity(
        skus,
        locations,
        orders,
        distance_metric=manhattan_distance,
        depot_location=(0, 0, 0)
    )

    assert slotting_plan["S1"] == "L1"
    assert slotting_plan["S2"] == "L2"
    # Zero-demand SKU should be placed in the remaining location
    assert slotting_plan["S3"] == "L3"
