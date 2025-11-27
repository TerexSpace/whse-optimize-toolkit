import pandas as pd
from typing import List
from .models import SKU, Location, Order, OrderLine, Warehouse

def load_generic_erp_data(skus_df: pd.DataFrame, locations_df: pd.DataFrame, orders_df: pd.DataFrame) -> Warehouse:
    """
    Loads warehouse data from generic pandas DataFrames.
    This is a simplified adapter assuming a specific CSV format.

    Args:
        skus_df: DataFrame with SKU master data (sku_id, name, ...).
        locations_df: DataFrame with location master data (loc_id, x, y, z, ...).
        orders_df: DataFrame with order data (order_id, sku_id, quantity, ...).

    Returns:
        A Warehouse object populated with the data.
    """
    skus = [
        SKU(
            sku_id=row['sku_id'],
            name=row.get('name', ''),
            weight=row.get('weight', 0.0),
            volume=row.get('volume', 0.0)
        )
        for _, row in skus_df.iterrows()
    ]
    sku_map = {sku.sku_id: sku for sku in skus}

    locations = []
    for _, row in locations_df.iterrows():
        loc_type = row.get('location_type', 'storage')
        if pd.isna(loc_type):
            loc_type = 'storage'
        loc_type = str(loc_type).lower()

        locations.append(
            Location(
                loc_id=row['loc_id'],
                coordinates=(row['x'], row['y'], row.get('z', 0)),
                capacity=row.get('capacity', 1.0),
                location_type=loc_type
            )
        )

    if not any(getattr(loc, "location_type", "").lower() == "depot" for loc in locations):
        raise ValueError("At least one location must be tagged as 'depot' (location_type='depot').")

    orders_grouped = orders_df.groupby('order_id')
    orders = []
    for order_id, group in orders_grouped:
        order_lines = [
            OrderLine(sku=sku_map[row['sku_id']], quantity=row['quantity'])
            for _, row in group.iterrows()
            if row['sku_id'] in sku_map
        ]
        if order_lines:
            orders.append(Order(order_id=str(order_id), order_lines=order_lines))

    return Warehouse(skus=skus, locations=locations, orders=orders)
