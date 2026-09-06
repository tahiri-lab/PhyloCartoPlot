# PhyloGeoPlot Unit Tests

This directory contains comprehensive unit tests for the PhyloGeoPlot package.

## Test Structure

```
tests/
├── conftest.py                      # Pytest configuration and shared fixtures
├── test_format_gbif_data.py         # Tests for geographic coordinate formatting
├── test_add_metadata.py             # Tests for metadata merging
├── test_build_phylogenetic_tree.py  # Tests for tree construction
├── test_tree_to_map_raster.py       # Tests for visualization
└── README.md                        # This file
```

## Test Coverage

### 1. `test_format_gbif_data.py`
Tests the `format_gbif()` function for formatting GBIF occurrence data:
- ✅ Output file creation
- ✅ Required columns present
- ✅ Correct specimen ID mapping
- ✅ NaN node name removal
- ✅ Coordinate preservation
- ✅ Duplicate handling
- ✅ Error handling for missing files
- ✅ Case sensitivity handling

### 2. `test_add_metadata.py`
Tests the `add_metadata()` function for merging trait values:
- ✅ Output file creation
- ✅ Required columns present
- ✅ Correct metadata merge
- ✅ Unmatched record removal
- ✅ NaN trait value handling
- ✅ Coordinate preservation
- ✅ Zero trait value handling
- ✅ Error handling
- ✅ Numeric trait value support

### 3. `test_build_phylogenetic_tree.py`
Tests the `build_tree()` function for phylogenetic tree construction:
- ✅ Tree file creation
- ✅ Valid Newick format output
- ✅ Correct tree structure
- ✅ Identical sequences handling
- ✅ Divergent sequences handling
- ✅ Branch structure validation
- ✅ Sequence name preservation
- ✅ Single sequence edge case
- ✅ Error handling for missing files
- ✅ Scalability with many sequences

### 4. `test_tree_to_map_raster.py`
Tests the `PhyloGeoPlotter` visualization class:
- ✅ Class initialization
- ✅ Data loading
- ✅ Auto vmin/vmax calculation
- ✅ Custom vmin/vmax parameters
- ✅ Error handling for missing files
- ✅ Color mapping
- ✅ Legend configuration
- ✅ Extent parameter handling
- ✅ Trait name customization
- ✅ Value-to-color mapping
- ✅ X-offset calculation
- ✅ Plot generation
- ✅ Figure saving (SVG/PNG)
- ✅ Figure closure
- ✅ Map mode operation
- ✅ Raster contrast methods

## Running the Tests

### Run all tests
```bash
pytest tests/
```

### Run tests with verbose output
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_format_gbif_data.py -v
```

### Run specific test class
```bash
pytest tests/test_add_metadata.py::TestAddMetadata -v
```

### Run specific test method
```bash
pytest tests/test_tree_to_map_raster.py::TestPhyloGeoPlotter::test_phylogeoplotter_initialization -v
```

### Run tests matching a pattern
```bash
pytest tests/ -k "metadata" -v
```

### Run tests excluding slow tests
```bash
pytest tests/ -m "not slow"
```

### Run with coverage report
```bash
pytest tests/ --cov=phylocartoplot --cov-report=html
```

## Test Dependencies

The tests require the following packages:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting (optional)
- `pandas` - Data manipulation
- `biopython` - Sequence analysis
- `matplotlib` - Visualization
- `cartopy` - Geographic plotting
- `rasterio` - Raster data handling
- `scikit-image` - Image processing

Install test dependencies:
```bash
pip install pytest pytest-cov pandas biopython matplotlib cartopy rasterio scikit-image
```

## Writing New Tests

### Template for a New Test Module

```python
"""
Unit tests for new_module.
Describe what the module does.
"""

import pytest
import tempfile
import os
from phylocartoplot.path.to.module import function_name


class TestFunctionName:
    """Test suite for function_name."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Your fixture code here
        pass

    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        # Your test code here
        pass

    def test_error_handling(self):
        """Test error handling."""
        # Your test code here
        pass
```

### Best Practices

1. **Use fixtures** for reusable test data
2. **Test edge cases** (empty input, single item, large datasets)
3. **Test error conditions** (missing files, invalid data)
4. **Keep tests independent** (no test should depend on another)
5. **Use descriptive names** for test functions
6. **Add docstrings** explaining what each test validates

## Continuous Integration

To run tests in CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ --cov=phylocartoplot --cov-report=xml
```

## Known Issues and Limitations

1. **Raster tests**: Some tests are skipped if proper GeoTIFF files are not available
2. **Visualization tests**: Plot rendering may require display backend on headless systems
3. **Performance tests**: Very large datasets may timeout in standard test runs

## Troubleshooting

### Import errors
Ensure the `phylocartoplot` package is properly installed:
```bash
pip install -e .
```

### Missing dependencies
Install all required packages:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

### Display backend errors
Set matplotlib to use non-interactive backend:
```bash
export MPLBACKEND=Agg
pytest tests/
```

## Contributing Tests

When adding new features:
1. Write tests first (TDD approach)
2. Implement the feature
3. Ensure all tests pass: `pytest tests/`
4. Check coverage: `pytest tests/ --cov=phylocartoplot`
5. Aim for >80% coverage on new code

## Contact

For questions about testing, please open an issue on the repository.
