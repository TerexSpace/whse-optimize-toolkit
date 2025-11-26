# WMS-OptLab: An open-source toolkit for optimization of ERP warehouse modules

WMS-OptLab is a Python research software package for optimizing and evaluating warehouse configurations, particularly those linked to Enterprise Resource Planning (ERP) and Warehouse Management Systems (WMS). It provides tools for storage location assignment (slotting), picker routing, and order batching, helping researchers and practitioners to design, analyze, and improve warehouse operations.

## Statement of Need

Modern warehouses are complex systems where efficiency is paramount. The layout of a warehouse, the assignment of products to storage locations (slotting), and the way orders are picked and batched have a profound impact on operational performance, particularly on travel distances, which can account for over 50% of a picker's time. While many commercial WMS offer some level of optimization, there is a need for open-source, extensible tools that allow researchers and practitioners to experiment with, evaluate, and develop novel optimization strategies. WMS-OptLab fills this gap by providing a modular, Python-based toolkit that integrates with standard ERP/WMS data exports and allows for what-if scenario analysis.

## Installation

You can install `wms-optlab` using pip:

```bash
pip install wms-optlab
```

## Minimal Example: Slotting Optimization

Here is a simple example of how to use `wms-optlab` to optimize storage locations based on product popularity (ABC analysis).

```python
import pandas as pd
from wms_optlab.data.adapters_erp import load_generic_erp_data
from wms_optlab.slotting.heuristics import assign_by_abc_popularity
from wms_optlab.layout.topology import manhattan_distance

# Load sample data
skus_df = pd.read_csv('examples/data/sample_skus.csv')
locations_df = pd.read_csv('examples/data/sample_locations.csv')
orders_df = pd.read_csv('examples/data/sample_orders.csv')

# Create warehouse model from data
warehouse = load_generic_erp_data(skus_df, locations_df, orders_df)

# Assign SKUs to locations based on popularity
# This heuristic assigns the most popular items to the best locations (closest to depot)
slotting_plan = assign_by_abc_popularity(
    warehouse.skus,
    warehouse.locations,
    warehouse.orders,
    distance_metric=manhattan_distance,
    depot_location=(0, 0)
)

# The 'slotting_plan' is a dictionary mapping SKU IDs to Location IDs
print("Slotting Plan:")
for sku_id, loc_id in list(slotting_plan.items())[:5]:
    print(f"  SKU {sku_id} -> Location {loc_id}")

# This is a simplified example. The toolkit also supports MILP-based optimization,
# picker routing analysis, and more complex scenarios.
```

WMS-OptLab is designed to be a flexible platform for research and education in warehouse logistics and optimization. We welcome contributions and feedback from the community.
