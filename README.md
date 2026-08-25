# PhyloCartoPlot

**PhyloCartoPlot** is a Python tool for phylogeographic visualization. It overlays phylogenetic trees on geographic raster maps, letting researchers explore the spatial distribution of evolutionary relationships and trait variation across species.

---

## Key Features

- Integrates GBIF occurrence data with phylogenetic trees
- Renders trees directly on GeoTIFF raster base maps (e.g. environmental data)
- Supports arbitrary traits via a generic `trait_value` column
- Works with any taxa, geographic region, or raster dataset
- Usable as a Python library (Jupyter) or command-line tool

---

## Requirements

- Python ≥ 3.8
- [Biopython](https://biopython.org/) – tree construction and parsing
- [Cartopy](https://scitools.org.uk/cartopy/) – geographic projections
- [Rasterio](https://rasterio.readthedocs.io/) – raster data I/O
- [Matplotlib](https://matplotlib.org/) – plotting
- pandas, numpy, scikit-image

---

## Installation

```bash
git clone https://github.com/tahiri-lab/PhyloCartoPlot.git
cd PhyloCartoPlot
pip install -e .
```

---

## Quick Start

### 1 – Format GBIF occurrence data

```bash
python -m phylocartoplot.preprocessing.format_gbif_data \
    examples/sample_data/use_case_1/gbif_coffea_ex3.csv \
    examples/sample_data/use_case_1/node_names.csv
```

### 2 – Add trait metadata

```bash
python -m phylocartoplot.preprocessing.add_metadata \
    examples/sample_data/use_case_1/gbif_coffea_ex3_formatted.csv \
    examples/sample_data/use_case_1/no_caffeine_nodes_w_specimen.csv
```

### 3 – Build phylogenetic tree

```bash
python -m phylocartoplot.preprocessing.build_phylogenetic_tree \
    sequences.fasta
```

### 4 – Visualize

```python
from phylocartoplot.visualisation.tree_to_map_raster import PhyloCartoPlotter

plotter = PhyloCartoPlotter(
    nwk_file="sequences_tree.nwk",
    gps_file="examples/sample_data/use_case_1/coords_w_caff.csv",
    offset_file="examples/sample_data/use_case_1/offsets_caff.csv",
    raster_file="enviro.tif",
    raster_band=1
)
plotter.plot()
plotter.save(output_dir="output")
```

Or use the interactive walkthrough notebooks in `examples/use_case_1/`.

---

## Documentation

Complete documentation for the PhyloCartoPlot workflow.

## Files

### 1. [PIPELINE.md](PIPELINE.md)
**Technical documentation of the entire workflow**

Explains:
- Module breakdown (what each script does)
- Input/output specifications
- Data flow diagrams
- Key functions and their purposes
- Customization points
- Troubleshooting guide

**Read this for:** Understanding how the pipeline works technically

---

### 2. [01_phylocartoplot_walkthrough.ipynb](examples/use_case_1/01_phylocartoplot_walkthrough.ipynb)
**Interactive step-by-step Jupyter notebook**

Walks through:
- Step 1: Format geographic coordinates
- Step 2: Add trait/metadata values
- Step 3: Build phylogenetic tree
- Step 4: Create visualization

**Read this for:** Hands-on learning, executing the workflow

#### Running the Notebook

```bash
# Navigate to docs folder
cd phylocartoplot/examples

# Start Jupyter
jupyter notebook

# Open: 01_phylocartoplot_walkthrough.ipynb
```

Or from project root:
```bash
jupyter notebook examples/use_case_1/01_phylocartoplot_walkthrough.ipynb
jupyter notebook examples/use_case_1/02_tree_to_map_raster_walkthrough.ipynb
```

---

## How to Use This Documentation

### For Quick Understanding
1. Read the main **README.md** 


### Tutorial
1. Open **[01_phylocartoplot_walkthrough.ipynb](examples/use_case_1/01_phylocartoplot_walkthrough.ipynb)** (examples folder)
2. Follow cells step-by-step
3. Execute and inspect outputs


---

## Notebook Features

Automatic path configuration
Step-by-step explanations
Data inspection and sampling
Error checking and reporting
Clear output messages
Next step instructions

---

---

## Quick Links

**New to PhyloCartoPlot?**
→ Start with **README.md**, then run the notebook in this folder

**Need technical details?**
→ Read [PIPELINE.md](PIPELINE.md) or check source code

**Want to understand the structure?**
→ See [STRUCTURE.txt](STRUCTURE.txt) in project root

**Ready to use the workflow?**
→ Run the notebook: `jupyter notebook examples/use_case_1/01_phylocartoplot_walkthrough.ipynb`

---

## Generality and Reusability

PhyloCartoPlot is designed as a parameterized, dataset-agnostic workflow. While the provided examples use *Coffea* species occurrence data and a WorldClim raster layer, the pipeline imposes no assumptions specific to that use case. Researchers can apply the tool to any combination of the following inputs:

- **Phylogenetic tree**: any Newick-formatted tree produced by standard inference tools
- **Taxa**: any group of organisms for which georeferenced occurrence records are available
- **Geographic region**: any spatial extent, limited only by the chosen raster layer coverage
- **Trait or metadata**: any continuous or categorical variable supplied via a `trait_value` column in the coordinate file
- **Raster base map**: any single-band or multi-band GeoTIFF (e.g., climate layers, land-cover, elevation)

To apply the workflow to a new dataset, it is sufficient to substitute the input files and adjust the column names and raster band index accordingly. No modifications to the source code are required for standard use cases.
