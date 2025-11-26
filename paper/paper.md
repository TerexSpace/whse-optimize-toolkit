---
title: 'WMS-OptLab: An open-source toolkit for optimization of ERP warehouse modules'
tags:
  - Python
  - warehouse management
  - optimization
  - ERP
  - operations research
authors:
  - name: Placeholder Author 1
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Placeholder Author 2
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2"
affiliations:
 - name: Placeholder University or Company
   index: 1
 - name: Another Placeholder Institution
   index: 2
date: 26 November 2025
bibliography: reference.lib
---

# Summary

WMS-OptLab is an open-source Python toolkit designed for the modeling, optimization, and evaluation of warehouse logistics operations, particularly those integrated with Enterprise Resource Planning (ERP) and Warehouse Management Systems (WMS). The package provides a modular framework for researchers and practitioners to analyze and improve key warehouse processes, including storage location assignment (slotting), order picking routes, and order batching. It features a layered architecture that separates data models, optimization algorithms, and evaluation metrics, promoting extensibility and interoperability. By offering both heuristic and exact optimization methods, along with tools for scenario analysis and visualization, WMS-OptLab serves as a valuable resource for academic research, education, and industrial engineering applications.

# Statement of Need

The efficiency of warehouse operations is a critical factor in supply chain performance. Picker travel time, in particular, can constitute over 50% of the total cost of order fulfillment [@de2007design]. Optimizing warehouse layout, slotting strategies, and routing policies can yield substantial cost savings and throughput improvements. While many commercial WMS packages include optimization modules, they often operate as "black boxes," limiting their use in research and customized industrial applications. There is a need for an open-source, well-documented, and extensible tool that allows for transparent experimentation with and development of new optimization algorithms. WMS-OptLab addresses this need by providing a Python-based, interoperable toolkit for warehouse optimization research and practice.

# State of the Field

The field of warehouse optimization is rich with academic literature on topics such as the traveling salesman problem (TSP) for picker routing, quadratic assignment problems (QAP) for slotting, and various heuristics for order batching [@roodbergen2001agv]. Several academic and a few open-source tools exist for specific sub-problems, but a comprehensive, integrated toolkit that connects these different facets of warehouse operations is less common. WMS-OptLab aims to bridge this gap by providing an easy-to-use library that can model the interdependencies between slotting, routing, and batching, and can be easily integrated with data from real-world ERP/WMS systems.

# Software Description

WMS-OptLab is structured into several core modules:

- **`data`**: Contains data models for core warehouse entities (e.g., `SKU`, `Location`, `Order`) and adapters for parsing data from common ERP/WMS export formats (e.g., CSV). It also includes synthetic data generators for research purposes.
- **`layout`**: Provides tools for representing warehouse geometry and topology, including the creation of graph-based network models for distance calculations using libraries like `networkx`.
- **`slotting`**: Implements algorithms for storage location assignment, ranging from simple ABC-popularity heuristics to MILP (Mixed-Integer Linear Programming) formulations that can be solved using backends like OR-Tools.
- **`routing`**: Includes common picker routing policies such as S-shape traversal and interfaces for TSP-based optimization of picking tours.
- **`batching`**: Offers heuristics and metaheuristics (e.g., Simulated Annealing) for grouping orders into efficient picking waves.
- **`evaluation`**: Provides a suite of performance metrics (e.g., total travel distance, workload balance) and reporting tools to compare different operational scenarios.
- **`solvers`**: An abstraction layer for optimization backends, with an initial implementation for Google's OR-Tools.

The software is written in Python 3.11+ with full type hinting and is designed with SOLID principles to ensure modularity and ease of extension.

# Illustrative Example

A common use case for WMS-OptLab is to evaluate a new slotting strategy. A user can load their current warehouse data (SKUs, locations, historical orders), and then use the `assign_by_abc_popularity` heuristic to generate a new slotting plan. The `calculate_expected_travel_distance` function can then be used to estimate the travel distance for both the current and proposed slotting plans, providing a quantitative basis for the decision. The results, including a visualization of the new layout, can be generated with a few lines of code, as shown in the accompanying example Jupyter notebooks.

# Availability and Reuse

WMS-OptLab is open-source software distributed under the MIT license. The source code, documentation, and examples are available on GitHub at [https://github.com/example/wms-optlab](https://github.com/example/wms-optlab). The package is designed for reuse and extension. Users can easily add new optimization algorithms, routing policies, or data adapters for different ERP systems.

# Acknowledgements

We acknowledge the contributions of the open-source community and the developers of the scientific Python stack, including `numpy`, `pandas`, `networkx`, and `ortools`, which form the foundation of this toolkit.

# References
