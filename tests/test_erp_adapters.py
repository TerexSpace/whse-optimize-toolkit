import pytest
import pandas as pd
from wms_optlab.data.adapters_erp import load_generic_erp_data

@pytest.fixture
def sample_dfs():
    """Provides sample DataFrames for testing ERP adapters."""
    skus_df = pd.DataFrame([
        {'sku_id': 'SKU1', 'name': 'Test SKU 1'},
        {'sku_id': 'SKU2', 'name': 'Test SKU 2'},
    ])
    locations_df = pd.DataFrame([
        {'loc_id': 'LOC1', 'x': 1, 'y': 2, 'z': 0, 'location_type': 'depot'},
        {'loc_id': 'LOC2', 'x': 3, 'y': 4, 'z': 0, 'location_type': 'storage'},
    ])
    orders_df = pd.DataFrame([
        {'order_id': 'ORD1', 'sku_id': 'SKU1', 'quantity': 10},
        {'order_id': 'ORD1', 'sku_id': 'SKU2', 'quantity': 5},
        {'order_id': 'ORD2', 'sku_id': 'SKU1', 'quantity': 8},
    ])
    return skus_df, locations_df, orders_df

def test_load_generic_erp_data(sample_dfs):
    """
    Tests the loading of data from generic pandas DataFrames.
    """
    skus_df, locations_df, orders_df = sample_dfs
    
    warehouse = load_generic_erp_data(skus_df, locations_df, orders_df)
    
    # Check if the counts are correct
    assert len(warehouse.skus) == 2
    assert len(warehouse.locations) == 2
    assert len(warehouse.orders) == 2 # Orders are grouped by order_id
    
    # Check a specific order to ensure it was parsed correctly
    order1 = next(o for o in warehouse.orders if o.order_id == 'ORD1')
    assert len(order1.order_lines) == 2
    
    # Check SKU and Location details
    sku1 = next(s for s in warehouse.skus if s.sku_id == 'SKU1')
    assert sku1.name == 'Test SKU 1'
    
    loc1 = next(l for l in warehouse.locations if l.loc_id == 'LOC1')
    assert loc1.coordinates == (1, 2, 0)


def test_load_generic_erp_data_requires_depot():
    skus_df = pd.DataFrame([{'sku_id': 'SKU1', 'name': 'Test SKU 1'}])
    # No depot tagged
    locations_df = pd.DataFrame([
        {'loc_id': 'LOC1', 'x': 1, 'y': 2, 'z': 0, 'location_type': 'storage'}
    ])
    orders_df = pd.DataFrame([{'order_id': 'ORD1', 'sku_id': 'SKU1', 'quantity': 1}])

    with pytest.raises(ValueError):
        load_generic_erp_data(skus_df, locations_df, orders_df)


def test_load_generic_erp_data_skips_empty_orders():
    skus_df = pd.DataFrame([
        {'sku_id': 'SKU1', 'name': 'Test SKU 1'},
    ])
    locations_df = pd.DataFrame([
        {'loc_id': 'DEPOT', 'x': 0, 'y': 0, 'z': 0, 'location_type': 'depot'}
    ])
    orders_df = pd.DataFrame([
        {'order_id': 'ORD1', 'sku_id': 'UNKNOWN', 'quantity': 1},  # will be dropped
        {'order_id': 'ORD2', 'sku_id': 'SKU1', 'quantity': 2},
    ])

    warehouse = load_generic_erp_data(skus_df, locations_df, orders_df)
    assert len(warehouse.orders) == 1
    assert warehouse.orders[0].order_id == 'ORD2'
