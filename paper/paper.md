---
title: 'WMS-OptLab: An open-source toolkit for optimization of ERP warehouse modules'
tags:
  - Python
  - warehouse management
  - supply chain
  - optimization
  - ERP
  - operations research
  - logistics
authors:
  - name: Almas Ospanov
    orcid: 0009-0004-3834-130X
    affiliation: "1"
affiliations:
  - name: Astana IT Univeristy, Kazakhstan
    index: 1
date: 27 November 2025
bibliography: paper.bib
repository: https://github.com/TerexSpace/whse-optimize-toolkit.git
DOI: https://doi.org/10.5281/zenodo.17739851
---

# Summary

WMS-OptLab is an open-source Python toolkit for modeling, optimizing, and evaluating warehouse logistics operations integrated with Enterprise Resource Planning (ERP) and Warehouse Management Systems (WMS). It provides a modular framework to analyze and improve storage location assignment (slotting), picker routing, order batching, and end-to-end what-if scenario analysis. The library exposes typed data models, heuristic and mathematical programming solvers, distance-aware layout builders, and evaluation utilities enabling researchers and practitioners to compare policies reproducibly. By offering interoperable components built atop the scientific Python stack [@harris2020array; @reback2020pandas; @hagberg2008exploring] and modern optimization backends [@ortools], WMS-OptLab lowers the barrier to experimenting with novel warehouse strategies and supports transparent, extensible research workflows.

# Statement of Need

Warehouse picker travel distances and associated labor costs can account for more than half of total order fulfillment effort [@de2007design]. While commercial WMS and ERP platforms typically include optimization capabilities, these modules are often proprietary, opaque, and difficult to adapt to specific operational constraints or research hypotheses. Researchers and industrial engineers therefore lack an open, extensible, peer-reviewable codebase for prototyping novel slotting strategies, routing heuristics, and batching policies that integrate directly with standard ERP/WMS data exports.

WMS-OptLab addresses this gap by providing a Python-based research toolkit with:

- **Data integration**: Direct CSV adapters for common ERP/WMS export formats, with validation to detect critical warehouse configurations (depot locations, SKU availability).
- **Modular optimization**: Pluggable implementations of established heuristics (ABC-popularity slotting, S-shape routing) and exact methods via mathematical programming, allowing researchers to mix and match components.
- **Reproducibility**: Deterministic algorithms with explicit parameter documentation, extensive unit test coverage, and example notebooks demonstrating typical workflows.
- **Extensibility**: Abstract base classes for solver backends, distance metrics, and routing policies enabling new research contributions without forking the codebase.
- **Scenario analysis**: Built-in what-if analysis framework to compare multiple configurations and quantify operational trade-offs.

# State of the Field

Warehouse optimization literature typically addresses routing, slotting, and batching as independent combinatorial problems: the Traveling Salesman Problem (TSP) for picker routing [@christofides1976worst], the Quadratic Assignment Problem (QAP) for slotting [@roodbergen2001agv], and bin-packing variants for batching. General-purpose optimization libraries (e.g., OR-Tools [@ortools], PuLP, Pyomo) and graph analysis toolkits (e.g., NetworkX [@hagberg2008exploring]) provide essential primitives for modeling and solving these problems.

However, an integrated, open-source toolkit that simultaneously addresses:
- **Data standardization**: CSV-to-warehouse-model conversion with ERP/WMS-specific validation,
- **Algorithm integration**: Side-by-side implementations of competing heuristics and exact solvers for benchmarking,
- **Evaluation infrastructure**: Distance calculations, route simulation, and KPI reporting,
- **Research reproducibility**: Deterministic outputs, comprehensive testing, and example workflows,

remains largely absent from the open-source landscape. WMS-OptLab consolidates these capabilities into a single, composable package designed for both research validation and practical experimentation in warehouse operations.

# Software Architecture

WMS-OptLab is organized into layered modules implementing distinct aspects of warehouse optimization:

**Data Layer** (`data/`)
- **Core models**: Immutable dataclasses (`SKU`, `Location`, `Order`, `Warehouse`) providing type-safe warehouse representations.
- **ERP adapters**: `load_generic_erp_data()` converts pandas DataFrames (standard ERP exports) into `Warehouse` objects with validation (depot detection, SKU filtering).
- **Data generators**: Synthetic SKU and order generators following realistic power-law demand distributions for reproducible research experiments.

**Layout & Geometry** (`layout/`)
- **Distance metrics**: Manhattan and Euclidean distance functions supporting 3D warehouse coordinates.
- **Topology**: `create_warehouse_graph()` builds networkx graphs of warehouse connectivity, enabling shortest-path calculations and TSP approximations.
- **Visualization**: 2D layout plotting with color-coded location utilization and route overlays for intuitive analysis.

**Optimization Components**
- **Slotting** (`slotting/`): ABC-popularity heuristic assigns high-demand SKUs to closest locations; MILP formulations via pluggable solver backends (currently OR-Tools CP-SAT) minimize total picker travel; evaluation computes expected travel distance under assumed demand.
- **Routing** (`routing/`): S-shape (serpentine) policy; Christofides TSP approximation for small pick sets; route simulation quantifies distance and pick counts per order.
- **Batching** (`batching/`): Due-date heuristics and simulated-annealing metaheuristics group orders under weight/volume constraints, minimizing batch cost.

**Evaluation & Reporting** (`evaluation/`)
- KPI calculations (workload per picker, congestion proxies, travel metrics).
- Text and Markdown report generation with pandas integration for export.

**Scenarios & What-If** (`scenarios/`)
- Multi-scenario comparison framework orchestrating slotting, routing, and evaluation to quantify operational trade-offs.

**Solver Abstraction** (`solvers/`)
- `OptimizationBackend` abstract interface enables swapping solver implementations (OR-Tools provided; extensible to other backends).

**Integration & Export** (`integration/`)
- WMS import file generation (CSV templates for slotting, SKU master, locations).
- Adapter stubs for external simulators (e.g., sme_erpsim).

The library targets Python 3.11+, employs type hints throughout, and uses frozen dataclasses and abstract base classes to facilitate extension (new routing policies, solver backends, distance metrics, ERP adapters) without modifying core code.

# Illustrative Example

A representative workflow ingests ERP CSV exports (SKUs, locations, historical orders), optimizes slotting, and compares travel performance:

```python
from wms_optlab.data.adapters_erp import load_generic_erp_data
from wms_optlab.slotting.heuristics import assign_by_abc_popularity
from wms_optlab.slotting.evaluation import calculate_expected_travel_distance
from wms_optlab.layout.geometry import manhattan_distance

# Load warehouse from ERP exports
warehouse = load_generic_erp_data(skus_df, locations_df, orders_df)

# Generate slotting plan via ABC-popularity heuristic
slotting_plan = assign_by_abc_popularity(
    warehouse.skus, warehouse.locations, warehouse.orders,
    distance_metric=manhattan_distance
)

# Evaluate expected picker travel distance
depot = next(l for l in warehouse.locations if l.location_type == 'depot')
graph = warehouse.get_graph()
travel_distance = calculate_expected_travel_distance(
    slotting_plan, warehouse.orders, graph, depot.loc_id
)
print(f"Expected travel: {travel_distance:.0f} units")
```

The repository includes Jupyter notebooks (`simple_warehouse_slotting.ipynb`, `current_vs_optimized_routing.ipynb`) and sample datasets (SKUs, locations, orders) demonstrating complete workflows and integration patterns.

# Testing and Quality Assurance

Test coverage includes:
- **Data layer**: CSV ingestion, validation logic, and synthetic data generation.
- **Optimization**: Heuristic correctness (ABC popularity), MILP constraint enforcement, routing algorithms (S-shape, TSP).
- **Integration**: End-to-end workflows combining multiple components.

Tests are executed via `pytest` (1148 lines of production code; 4 test modules) with continuous integration (GitHub Actions) running the full suite on each commit to main and pull request. Example notebooks serve as integration smoke tests.

# Applications and Research Impact

WMS-OptLab enables:
- **Reproducible research**: Deterministic, open-source implementations allow peer review and methodological validation.
- **Comparative studies**: Side-by-side heuristic and exact method evaluation for warehouse policies.
- **Practitioner validation**: CSV adapters allow practitioners to evaluate proposals on real data before deployment.
- **Curriculum support**: Module structure and type hints make the toolkit suitable for teaching supply chain optimization and software engineering in operations research.

# Availability and Reuse

WMS-OptLab is released under the permissive MIT License, conforming to the Open Source Initiative Definition [@osd]. Source code, comprehensive API documentation, example notebooks, and sample datasets are hosted at [https://github.com/TerexSpace/whse-optimize-toolkit](https://github.com/TerexSpace/whse-optimize-toolkit) under a public GitHub repository with an open issue tracker.

Users can extend the toolkit with custom routing policies, solver backends (via `OptimizationBackend` interface), distance metrics, or ERP adapters while reusing core data models, evaluation utilities, and scenario infrastructure. A `CONTRIBUTING.md` document outlines contribution guidelines and development workflows.

# Acknowledgements

We acknowledge the scientific Python community and the developers of foundational libraries: NumPy [@harris2020array], Pandas [@reback2020pandas], NetworkX [@hagberg2008exploring], and Google OR-Tools [@ortools], upon which WMS-OptLab is built.

# References
