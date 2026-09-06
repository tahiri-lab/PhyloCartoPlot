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

## Tests

Run the test suite with:

```bash
python -m pytest -v
```

## License

PhyloGeoPlot is distributed under the [MIT License](https://github.com/tahiri-lab/PhyloGeoPlot/blob/main/LICENSE).

## Author

**Caroline Fortier**

## Citation

Citation information will be added here following publication.
