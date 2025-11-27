# Contributing to WMS-OptLab

Thank you for your interest in contributing to WMS-OptLab! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/TerexSpace/whse-optimize-toolkit.git
   cd whse-optimize-toolkit
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   ```

3. Install in development mode with test dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e .[dev]
   ```

4. Verify the installation:
   ```bash
   PYTHONPATH=src pytest -q
   ```

## Code Contributions

### Types of Contributions

We welcome contributions in several areas:

- **Bug fixes**: Report and fix issues in existing functionality
- **New routing policies**: Implement additional picking strategies (e.g., aisle-based, zone-based)
- **Solver backends**: Add support for new optimization backends (e.g., Pyomo, PuLP)
- **Distance metrics**: Contribute custom warehouse distance calculations (e.g., time-based metrics)
- **ERP/WMS adapters**: Support new data import formats and systems
- **Tests**: Expand test coverage and edge case handling
- **Documentation**: Improve README, API documentation, and examples
- **Performance improvements**: Optimize algorithms and data structures

### Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures and return types
- Keep function and module docstrings clear and concise
- Use meaningful variable names
- Maintain line length around 100 characters (soft limit)

Example:
```python
from typing import Dict, List
from wms_optlab.data.models import SKU, Location

def assign_by_custom_heuristic(
    skus: List[SKU],
    locations: List[Location],
    demand_data: Dict[str, int],
) -> Dict[str, str]:
    """
    Assign SKUs to locations using a custom heuristic.

    Args:
        skus: List of SKU objects
        locations: List of warehouse locations
        demand_data: Dictionary mapping SKU ID to demand count

    Returns:
        Dictionary mapping SKU ID to assigned Location ID
    """
    # Implementation here
    pass
```

### Testing Requirements

- Write tests for all new functionality
- Ensure all existing tests pass before submitting
- Target >80% code coverage for new modules
- Use `pytest` fixtures for test data setup

Running tests locally:
```bash
# Run all tests with coverage
PYTHONPATH=src pytest --cov=wms_optlab

# Run specific test file
PYTHONPATH=src pytest tests/test_my_feature.py -v

# Run with verbose output
PYTHONPATH=src pytest -vv
```

### Documentation

- Add docstrings to all public functions and classes
- Include type hints in function signatures
- Update README.md for user-facing changes
- Add example code in docstrings for complex functions
- Update paper.md if your contribution significantly changes the software scope

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** and commit regularly:
   ```bash
   git add src/wms_optlab/my_module.py
   git commit -m "Add new routing policy: zone-based picking"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

4. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to any related issues (e.g., "Fixes #42")
   - Test results demonstrating the fix/feature works
   - Updated documentation if applicable

### Commit Message Guidelines

Use clear, descriptive commit messages:

```
[Category] Brief summary (50 chars max)

More detailed explanation (if needed) explaining why this change
is necessary and what problem it solves. Keep lines to ~72 characters.

References: #123
```

Categories: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Examples:
- `feat: implement zone-based routing policy`
- `fix: correct calculation in ABC popularity heuristic`
- `docs: add example notebook for scenario analysis`
- `test: expand batching metaheuristic edge cases`

## Issue Reporting

Before reporting an issue, please:

1. Check existing issues to avoid duplicates
2. Use a clear, descriptive title
3. Provide a minimal reproducible example:
   ```python
   from wms_optlab.data.adapters_erp import load_generic_erp_data

   warehouse = load_generic_erp_data(skus_df, locations_df, orders_df)
   # Error occurs here...
   ```
4. Include your Python version and OS
5. Attach relevant data files or stack traces

## Extending WMS-OptLab

### Adding a New Routing Policy

1. Create implementation in `src/wms_optlab/routing/policies.py`:
   ```python
   def get_zone_based_route(
       depot_location: Location,
       pick_locations: List[Location],
       zones: Dict[str, List[Location]],
   ) -> List[Location]:
       """Zone-based picking strategy."""
       # Implementation
       return route
   ```

2. Add tests in `tests/test_routing_policies.py`

3. Update `src/wms_optlab/routing/simulation_interface.py` to support the new policy

4. Document usage in README.md

### Adding a New Solver Backend

1. Implement `OptimizationBackend` interface in `src/wms_optlab/solvers/`:
   ```python
   from wms_optlab.solvers.interfaces import OptimizationBackend

   class MyNewBackend(OptimizationBackend):
       def add_binary_var(self, name: str) -> str:
           # Implementation
           pass
       # Implement all abstract methods
   ```

2. Add tests in `tests/test_solvers.py`

3. Update documentation with backend usage example

### Adding an ERP Adapter

1. Create adapter in `src/wms_optlab/data/adapters_erp.py`:
   ```python
   def load_sap_erp_data(skus_df, locations_df, orders_df):
       """Convert SAP ERP exports to Warehouse model."""
       # SAP-specific parsing logic
       return load_generic_erp_data(skus_df, locations_df, orders_df)
   ```

2. Add tests validating adapter-specific transformations

3. Include example data and usage in `examples/`

## Code Review Process

All contributions undergo peer review:

- Maintainers will review for correctness, style, and alignment with project goals
- Address feedback respectfully and iteratively
- Once approved, your changes will be merged to main

## Recognition

Contributors are acknowledged in:
- CITATION.cff (for substantial contributions)
- Paper.md acknowledgements section (as appropriate)
- GitHub contributors page

## Questions?

Open an issue with the `question` label or contact the maintainers directly.

Thank you for contributing to WMS-OptLab!
