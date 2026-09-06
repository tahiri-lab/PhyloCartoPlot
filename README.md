# PhyloCartoPlot

**PhyloCartoPlot** is a Python package for linked visualization of phylogenetic trees and geographic occurrence data. It connects taxa displayed in a phylogenetic tree to their geographic locations on a map and can optionally incorporate raster environmental data.

The package is intended for reproducible phylogeographic and biodiversity visualization workflows using standard formats such as Newick trees, CSV occurrence data, and GeoTIFF rasters.

![PhyloCartoPlot tree-to-map visualization](https://github.com/tahiri-lab/PhyloCartoPlot/blob/main/images/tree2map_caff_raster.png?raw=1)


## Features

PhyloCartoPlot provides tools to:

- format GBIF occurrence data for downstream analysis;
- attach trait or environmental metadata to occurrence records;
- construct phylogenetic trees from aligned sequence data;
- display phylogenetic trees alongside geographic occurrence maps;
- link tree tips to mapped occurrence points;
- apply consistent trait-based color encoding across tree and map components;
- optionally display environmental raster data;
- export publication-quality figures in multiple formats.

## Installation

PhyloCartoPlot is available from PyPI:

```bash
pip install phylocartoplot
```

PhyloCartoPlot requires **Python >= 3.10**.

Main dependencies include:

- Biopython
- Cartopy
- Rasterio
- Matplotlib
- Pandas
- NumPy
- scikit-image

These dependencies are installed automatically by `pip`.

## Basic workflow

A typical workflow consists of preparing occurrence data, optionally adding metadata, building or loading a phylogenetic tree, and generating the linked tree-map visualization.

### 1 – Format GBIF occurrence data

```bash
python -m phylocartoplot.preprocessing.format_gbif_data \
    gbif_occurrences.csv \
    node_names.csv
```

### 2 – Add trait metadata

```bash
python -m phylocartoplot.preprocessing.add_metadata \
    gbif_occurrences_formatted.csv \
    trait_metadata.csv
```

### 3 – Build a phylogenetic tree

```bash
python -m phylocartoplot.preprocessing.build_phylogenetic_tree \
    sequences.fasta
```

### 4 – Visualize the phylogeny and geographic data

```python
from phylocartoplot.visualisation.tree_to_map_raster import PhyloCartoPlotter

plotter = PhyloCartoPlotter(
    nwk_file="sequences_tree.nwk",
    gps_file="coordinates.csv",
    offset_file="offsets.csv",
    raster_file="enviro.tif",
    raster_band=1,
)

plotter.plot()
plotter.save(output_dir="output")
```

Raster input is optional. PhyloCartoPlot can also generate linked phylogenetic and geographic visualizations without an environmental raster.

## Input data

PhyloCartoPlot works with standard biological and geospatial data formats.

### Phylogenetic tree

Phylogenetic trees are supplied in **Newick (`.nwk`)** format.

Trees may be generated externally or created from aligned sequence data using:

```bash
python -m phylocartoplot.preprocessing.build_phylogenetic_tree sequences.fasta
```

### Geographic occurrence data

Geographic occurrence data are supplied as CSV files containing taxon or specimen identifiers and geographic coordinates.

GBIF occurrence exports can be reformatted using:

```bash
python -m phylocartoplot.preprocessing.format_gbif_data
```

### Trait metadata

Additional quantitative traits can be associated with occurrence records and represented using a continuous color scale in the visualization.

Examples include:

- caffeine concentration;
- environmental measurements;
- morphological traits;
- ecological measurements.

### Raster data

Environmental raster data can be supplied as GeoTIFF files.

Examples include:

- elevation;
- precipitation;
- temperature;
- vegetation indices;
- other continuous environmental variables.

Raster support is provided through Rasterio.

## Trait-based visualization

PhyloCartoPlot can apply the same trait-based color encoding to phylogenetic elements, mapped occurrences, and connecting lines.

![PhyloCartoPlot tree-to-map visualization](https://github.com/tahiri-lab/PhyloCartoPlot/blob/main/images/tree2map_caff_raster.png?raw=1)


This allows phylogenetic relationships, geographic distributions, and quantitative traits to be examined together within a single figure.

## Example datasets and walkthroughs

Complete walkthrough notebooks and sample datasets are available in the GitHub repository:

[PhyloCartoPlot GitHub repository](https://github.com/tahiri-lab/PhyloCartoPlot)

The repository currently includes two principal use cases.

### Use case 1 – Malagasy *Coffea*

This example links a phylogenetic tree of Malagasy *Coffea* taxa to specimen occurrence records in Madagascar.

Trait values such as caffeine concentration can be represented using a continuous color scale, and environmental raster layers can be included in the geographic panel.

Example notebooks:

- [PhyloCartoPlot walkthrough](https://github.com/tahiri-lab/PhyloCartoPlot/blob/main/examples/use_case_1/01_phylocartoplot_walkthrough.ipynb)
- [Tree-to-map raster walkthrough](https://github.com/tahiri-lab/PhyloCartoPlot/blob/main/examples/use_case_1/02_tree_to_map_raster_walkthrough.ipynb)

Sample data are available under:

```text
examples/sample_data/use_case_1/
```

### Use case 2 – North Atlantic Cumacea

This example illustrates linked phylogenetic and geographic visualization for Cumacea taxa distributed across North Atlantic regions.

Example notebook:

- [Cumacea tree-to-map walkthrough](https://github.com/tahiri-lab/PhyloCartoPlot/blob/main/examples/use_case_2/03_tree_to_map_walkthrough.ipynb)

![Tree-to-map Cumacea in North Atlantic](https://github.com/tahiri-lab/PhyloCartoPlot/blob/main/images/tree2map_cumacea.png?raw=1)

Sample data are available under:

```text
examples/sample_data/use_case_2/
```

## Running the repository examples

The example datasets and notebooks are maintained in the GitHub repository and are not bundled into the PyPI installation.

To reproduce the examples, clone the repository:

```bash
git clone https://github.com/tahiri-lab/PhyloCartoPlot.git
cd PhyloCartoPlot
```

Install the package in editable mode with the development dependencies:

```bash
pip install -e ".[dev]"
```

The example files can then be accessed under:

```text
examples/
├── sample_data/
│   ├── use_case_1/
│   └── use_case_2/
├── use_case_1/
└── use_case_2/
```

For example:

```bash
python -m phylocartoplot.preprocessing.format_gbif_data \
    examples/sample_data/use_case_1/gbif_coffea_ex3.csv \
    examples/sample_data/use_case_1/node_names.csv
```

## Output

PhyloCartoPlot can generate linked phylogeny-map figures in formats suitable for exploratory analysis and publication.

Depending on the selected options, output may include:

- SVG;
- PNG;
- PDF.

Output files can be written to a user-specified directory:

```python
plotter.save(output_dir="output")
```

## Development installation

For development or contribution, clone the repository:

```bash
git clone https://github.com/tahiri-lab/PhyloCartoPlot.git
cd PhyloCartoPlot
```

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Development dependencies currently include:

- pytest
- Jupyter

## Tests

The project includes automated tests for preprocessing and visualization components.

Run the test suite with:

```bash
python -m pytest -v
```

The current test suite covers:

- metadata integration;
- GBIF data formatting;
- phylogenetic tree construction;
- visualization initialization;
- trait color mapping;
- raster configuration;
- plotting;
- figure export.

## Building and publishing

PhyloCartoPlot uses a standard `pyproject.toml` (PEP 517/518/621) build configuration
with `setuptools` as the build backend. To build distributable artifacts locally:

```bash
python -m pip install --upgrade build twine
python -m build
```

This produces a source distribution (`.tar.gz`) and a wheel (`.whl`) in the `dist/`
directory. Validate the distribution metadata before uploading:

```bash
twine check dist/*
```

To upload to [TestPyPI](https://test.pypi.org/) first (recommended for verifying a
release before publishing it for real):

```bash
twine upload --repository testpypi dist/*
```

Then install from TestPyPI to confirm everything works:

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps phylocartoplot
```

Once verified, upload to [PyPI](https://pypi.org/):

```bash
twine upload dist/*
```

Releases can also be published automatically via the `.github/workflows/publish.yml`
GitHub Actions workflow, which runs on every published GitHub Release and uses
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so no
PyPI API token needs to be stored as a secret. See the comments at the top of that
workflow file for the exact PyPI project configuration steps required before the
first release.

## Project structure

The main Python package is organized as:

```text
phylocartoplot/
├── preprocessing/
│   ├── add_metadata.py
│   ├── build_phylogenetic_tree.py
│   ├── format_gbif_data.py
│   └── prepare_data.py
│
└── visualisation/
    └── tree_to_map_raster.py
```

Example data, notebooks, workflow documentation, and figures are maintained separately in the GitHub repository.

## Repository resources

Additional project resources are available on GitHub:

- [Source code](https://github.com/tahiri-lab/PhyloCartoPlot)
- [Example notebooks](https://github.com/tahiri-lab/PhyloCartoPlot/tree/main/examples)
- [Sample datasets](https://github.com/tahiri-lab/PhyloCartoPlot/tree/main/examples/sample_data)
- [Workflow documentation](https://github.com/tahiri-lab/PhyloCartoPlot/tree/main/workflow/docs)


## Citation

If you use PhyloCartoPlot in a scientific publication, please cite the associated software publication when available.

Citation information will be added here following publication.

## License

PhyloCartoPlot is distributed under the **MIT License**.

See the [LICENSE](https://github.com/tahiri-lab/PhyloCartoPlot/blob/main/LICENSE) file for details.

## Author

**Caroline Fortier**

## Repository

Source code, examples, issues, and development history are available at:

[https://github.com/tahiri-lab/PhyloCartoPlot](https://github.com/tahiri-lab/PhyloCartoPlot)

## Contributing

Bug reports, feature requests, and contributions are welcome through the GitHub repository.
