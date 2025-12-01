# WMS-OptLab: An open-source toolkit for optimization of ERP warehouse modules

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/TerexSpace/whse-optimize-toolkit/workflows/CI/badge.svg)](https://github.com/TerexSpace/whse-optimize-toolkit/actions)

WMS-OptLab is a Python research software package for optimizing and evaluating warehouse operations, particularly those integrated with Enterprise Resource Planning (ERP) and Warehouse Management Systems (WMS). It provides research-grade tools for storage location assignment (slotting), picker routing, order batching, and what-if scenario analysis, enabling researchers and practitioners to design, evaluate, and improve warehouse configurations reproducibly.

## Overview

Warehouse picker travel distances and labor costs can represent over 50% of total fulfillment effort, yet most commercial WMS optimization modules remain proprietary and opaque. WMS-OptLab addresses this gap by providing:

- **Modular optimization**: ABC popularity heuristics, mathematical programming (MILP), and metaheuristic methods for slotting, routing, and batching
- **Data integration**: Direct CSV adapters for standard ERP/WMS exports with built-in validation
- **Reproducible research**: Deterministic algorithms with comprehensive testing and example workflows
- **Extensibility**: Pluggable solver backends, distance metrics, and routing policies via abstract interfaces
- **Scenario analysis**: What-if framework to compare multiple warehouse configurations and quantify trade-offs

## Installation

Install from source (Python 3.11+):

```bash
git clone https://github.com/TerexSpace/whse-optimize-toolkit.git
cd whse-optimize-toolkit
python -m venv .venv && .\.venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
pip install --upgrade pip
pip install -e .[dev]  # Installs main dependencies plus those for testing and examples
```

## Minimal Example: Slotting Optimization

Here is a simple example of how to use `wms-optlab` to optimize storage locations based on product popularity (ABC analysis).

```python
import pandas as pd
from wms_optlab.data.adapters_erp import load_generic_erp_data
from wms_optlab.slotting.heuristics import assign_by_abc_popularity
from wms_optlab.layout.geometry import manhattan_distance

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
    depot_location=(0, 0, 0)
)

# The 'slotting_plan' is a dictionary mapping SKU IDs to Location IDs
print("Slotting Plan:")
for sku_id, loc_id in list(slotting_plan.items())[:5]:
    print(f"  SKU {sku_id} -> Location {loc_id}")

# This is a simplified example. The toolkit also supports MILP-based optimization,
# picker routing analysis, and more complex scenarios.
```

## Testing

Run the test suite with:

```bash
PYTHONPATH=src python -m pytest -q
```

## Features

### Slotting (Storage Location Assignment)
- **ABC Popularity Heuristic**: Fast, deterministic assignment based on SKU demand
- **MILP Optimization**: Exact methods via pluggable solver backend (OR-Tools CP-SAT)
- **Distance Evaluation**: Calculate expected picker travel for a slotting plan

### Routing
- **S-Shape Policy**: Classic serpentine picking strategy
- **TSP Approximation**: Christofides algorithm for small pick sets
- **Route Simulation**: Quantify distance and picks per order

### Batching
- **Heuristics**: Due-date grouping with capacity constraints
- **Metaheuristics**: Simulated annealing for large problem instances
- **Constraint Validation**: Weight and volume limit checking

### Analysis & Scenarios
- **What-if Comparison**: Evaluate multiple warehouse configurations
- **KPI Reporting**: Workload, congestion, distance metrics
- **Visualization**: 2D warehouse layout plots with route overlays

## API Documentation

### Core Data Models

```python
from wms_optlab.data.models import SKU, Location, Order, Warehouse

# Define SKU
sku = SKU(sku_id="SKU-001", name="Widget", weight=1.5, volume=0.02)

# Define Location
loc = Location(loc_id="A1-1", x=10, y=20, z=0, location_type="storage", capacity=100)

# Define Order
order = Order(order_id="ORD-001", lines=[OrderLine(sku_id="SKU-001", quantity=5)])

# Create Warehouse
warehouse = Warehouse(skus=[sku, ...], locations=[loc, ...], orders=[order, ...])
```

### Optimization Functions

```python
# Slotting
from wms_optlab.slotting.heuristics import assign_by_abc_popularity
slotting_plan = assign_by_abc_popularity(warehouse.skus, warehouse.locations, warehouse.orders)

# Routing
from wms_optlab.routing.policies import get_s_shape_route
route = get_s_shape_route(depot, pick_locations)

# Batching
from wms_optlab.batching.metaheuristics import simulated_annealing_batching
batches = simulated_annealing_batching(orders, max_weight=1000, iterations=1000)

# Scenarios
from wms_optlab.scenarios.what_if import run_what_if_analysis
results = run_what_if_analysis(warehouse, orders, scenarios)
```

See the [examples](examples/) directory for complete notebooks and detailed usage patterns.

## Documentation Structure

- **API docs**: Type hints and docstrings in source code
- **Examples**: [Jupyter notebooks](examples/) demonstrating workflows
- **Tests**: [Unit tests](tests/) serve as usage examples
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

## Quality Assurance

- **Unit Tests**: 4 test modules covering data, optimization, routing, and batching (run via `pytest`)
- **Continuous Integration**: GitHub Actions runs full test suite on every commit and PR
- **Code Coverage**: Target >80% coverage for new modules
- **Documentation**: Comprehensive docstrings and examples in repository

## Citation

If you use WMS-OptLab in academic work, please cite:

```bibtex
@software{wmsoptlab2025,
  title={WMS-OptLab: An open-source toolkit for optimization of ERP warehouse modules},
  author={Ospanov, Almas},
  year={2025},
  url={https://github.com/TerexSpace/whse-optimize-toolkit}
}
```

A `CITATION.cff` file is provided in the repository for automatic citation generation.

## Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup and coding guidelines
- Testing requirements
- Pull request process
- Ideas for contributions (new routing policies, solver backends, adapters, tests, documentation)

## Support

- **Issues**: Use the [GitHub Issues](https://github.com/TerexSpace/whse-optimize-toolkit/issues) tracker for bug reports and feature requests
- **Discussions**: Open an issue with the `question` label for usage questions
- **License**: MIT (see [LICENSE](LICENSE) file)

WMS-OptLab is designed to be a flexible, extensible platform for research and education in warehouse logistics optimization. We invite researchers, engineers, and students to contribute and use this toolkit in their work.
