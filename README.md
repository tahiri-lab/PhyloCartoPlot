# PhyloGeoPlot

**PhyloGeoPlot** is a Python package for linked visualization of phylogenetic trees and geographic occurrence data. It connects taxa represented in a phylogenetic tree to their geographic locations on a map and can optionally incorporate environmental raster data and quantitative trait information.

![PhyloGeoPlot example](https://raw.githubusercontent.com/tahiri-lab/PhyloGeoPlot/main/images/tree2map_caff_raster_readme.png)

## Installation

Install from PyPI:

```bash
pip install phylogeoplot
```

PhyloGeoPlot requires **Python 3.10 or later**.

## Dependencies

Main dependencies are installed automatically with `pip`:

- Biopython
- Cartopy
- Rasterio
- Matplotlib
- Pandas
- NumPy
- scikit-image

## Basic usage

```python
from phylogeoplot.visualisation.tree_to_map_raster import PhyloGeoPlotter

plotter = PhyloGeoPlotter(
    nwk_file="sequences_tree.nwk",
    gps_file="coordinates.csv",
    offset_file="offsets.csv",
    raster_file="environment.tif",
    raster_band=1,
)

plotter.plot()
plotter.save(output_dir="output")
```

Raster input is optional.

## Documentation

Detailed documentation is available in the [PhyloGeoPlot Wiki](https://github.com/tahiri-lab/PhyloGeoPlot/wiki).

Useful pages:

- [Installation](https://github.com/tahiri-lab/PhyloGeoPlot/wiki/Installation)
- [Input Data](https://github.com/tahiri-lab/PhyloGeoPlot/wiki/Input-Data)
- [Preprocessing](https://github.com/tahiri-lab/PhyloGeoPlot/wiki/Preprocessing)
- [Visualization](https://github.com/tahiri-lab/PhyloGeoPlot/wiki/Visualization)
- [Examples](https://github.com/tahiri-lab/PhyloGeoPlot/wiki/Examples)
- [Testing and Development](https://github.com/tahiri-lab/PhyloGeoPlot/wiki/Testing-and-Development)
- [Troubleshooting](https://github.com/tahiri-lab/PhyloGeoPlot/wiki/Troubleshooting)
- [Citation](https://github.com/tahiri-lab/PhyloGeoPlot/wiki/Citation)

## Examples

The repository contains sample datasets and walkthrough notebooks for:

- Malagasy *Coffea*
- North Atlantic Cumacea

See the [Examples](https://github.com/tahiri-lab/PhyloGeoPlot/wiki/Examples) page for details.

## Source code

Source code and example data are available on GitHub:

https://github.com/tahiri-lab/PhyloGeoPlot

## Tests

Run the test suite with:

```bash
python -m pytest -v
```

## Release process

PhyloGeoPlot is packaged with a standard `pyproject.toml` (setuptools backend).

Build the distribution locally:

```bash
python -m pip install --upgrade pip build twine
python -m build
```

Validate the built artifacts before uploading:

```bash
python -m twine check dist/*
```

Upload to TestPyPI first to confirm everything works:

```bash
python -m twine upload --repository testpypi dist/*
```

Once verified, upload to PyPI:

```bash
python -m twine upload dist/*
```

Releases are also published automatically by the `.github/workflows/publish.yml`
GitHub Actions workflow whenever a GitHub Release is published, or manually via
`workflow_dispatch` (which lets you choose the `pypi` or `testpypi` target).
This workflow uses [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API token needs to be stored as a repository secret.

**Before the first release**, `phylogeoplot` does not exist as a project on
PyPI/TestPyPI yet, so the trusted publisher cannot be added from a project's
settings page. Instead, register a **pending publisher** from the
account-level publishing page:

- PyPI: <https://pypi.org/manage/account/publishing/>
- TestPyPI: <https://test.pypi.org/manage/account/publishing/>

using these values (matching the workflow exactly, or PyPI will reject the
publish with an `invalid-publisher` error):

| Field                  | Value              |
| ---------------------- | ------------------- |
| PyPI project name      | `phylogeoplot`       |
| Owner                  | `tahiri-lab`         |
| Repository name        | `PhyloGeoPlot`       |
| Workflow filename      | `publish.yml`        |
| Environment name       | `pypi` (PyPI) / `testpypi` (TestPyPI) |

Once the project has been published for the first time, PyPI automatically
converts the pending publisher into a regular one, visible in the project's
own "Publishing" settings.

## License

PhyloGeoPlot is distributed under the [MIT License](https://github.com/tahiri-lab/PhyloGeoPlot/blob/main/LICENSE).

## Author

**Caroline Fortier**

## Citation

Citation information will be added here following publication.
