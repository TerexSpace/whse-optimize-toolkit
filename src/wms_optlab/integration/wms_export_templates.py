import pandas as pd
from typing import List, Dict
from ..data.models import SKU, Location, Order

def generate_wms_slotting_import_file(slotting_plan: Dict[str, str]) -> pd.DataFrame:
    """
    Generates a DataFrame in a format suitable for importing a new slotting plan
    into a Warehouse Management System (WMS).

    Args:
        slotting_plan: A dictionary mapping SKU IDs to Location IDs.

    Returns:
        A pandas DataFrame with 'sku_id' and 'location_id' columns.
    """
    df = pd.DataFrame(list(slotting_plan.items()), columns=['sku_id', 'location_id'])
    return df

def get_order_export_template() -> pd.DataFrame:
    """Returns a template DataFrame for order data exports."""
    return pd.DataFrame(columns=['order_id', 'sku_id', 'quantity', 'due_date'])

def get_sku_master_export_template() -> pd.DataFrame:
    """Returns a template DataFrame for SKU master data exports."""
    return pd.DataFrame(columns=['sku_id', 'name', 'description', 'weight', 'volume'])

def get_location_master_export_template() -> pd.DataFrame:
    """Returns a template DataFrame for location master data exports."""
    return pd.DataFrame(columns=['loc_id', 'x', 'y', 'z', 'capacity', 'location_type'])
