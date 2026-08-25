# PhyloCartoPlot – Technical Pipeline Documentation

This document describes the complete workflow of PhyloCartoPlot, from raw GBIF occurrence data to a phylogenetic tree overlaid on a geographic raster map.

---

## Overview

```
GBIF occurrence CSV  ──► format_gbif_data.py  ──► formatted CSV
                                                        │
Trait/metadata CSV   ──────────────────────────► add_metadata.py  ──► annotated CSV
                                                                              │
FASTA alignment      ──► build_phylogenetic_tree.py  ──► NWK tree file       │
                                                              │               │
                                              ┌───────────────┘               │
                                              ▼                               ▼
                                    tree_to_map_raster.py  ◄── offsets CSV ──┘
                                              │
                                              ▼
                                    PNG / PDF map output
```

---

## Step 1 – Format GBIF Data (`format_gbif_data.py`)

**Purpose:** Match GBIF specimen IDs with phylogenetic node names and extract geographic coordinates.

### Input
| File | Required columns |
|------|-----------------|
| GBIF occurrence CSV | `specimen_id`, `longitude`, `latitude` |
| Node names CSV | `node_name` |

### Processing
1. Reads GBIF occurrence data and node names.
2. Extracts a "key" from each `specimen_id` by stripping trailing numeric suffixes (`_\d+`).
3. Strips leading `C_` prefix and trailing alphanumeric suffixes from `node_name` to produce a matching key.
4. Maps node names onto GBIF rows via the shared key.

### Output
A CSV file with the suffix `_formatted` appended to the input filename, containing:
`specimen_id`, `longitude`, `latitude`, `node_name`

### Example
```bash
python -m phylocartoplot.preprocessing.format_gbif_data \
    gbif_coffea_ex3.csv node_names.csv
# produces: gbif_coffea_ex3_formatted.csv
```

---

## Step 2 – Add Metadata / Traits (`add_metadata.py`)

**Purpose:** Merge trait or metadata values (e.g. biochemical presence/absence) into the formatted occurrence data.

### Input
| File | Required columns |
|------|-----------------|
| Formatted GBIF CSV (from Step 1) | `specimen_id`, `longitude`, `latitude` |
| Metadata/traits CSV | `Species_name`, `trait_value` |

### Processing
1. Reads both files.
2. Left-joins metadata on `specimen_id` ↔ `Species_name`.
3. Drops rows where `trait_value` is missing.
4. Retains only: `specimen_id`, `longitude`, `latitude`, `trait_value`.

### Output
A CSV file with `_formatted` replaced by `_w_metadata` in the filename.

### Example
```bash
python -m phylocartoplot.preprocessing.add_metadata \
    gbif_coffea_ex3_formatted.csv no_caffeine_nodes_w_specimen.csv
# produces: gbif_coffea_ex3_w_metadata.csv
```

### Customisation
The `trait_value` column is generic – it can represent any discrete or continuous trait (presence/absence, colour category, quantitative measurement, etc.).

---

## Step 3 – Build Phylogenetic Tree (`build_phylogenetic_tree.py`)

**Purpose:** Construct a Neighbour-Joining phylogenetic tree from a multiple-sequence FASTA alignment.

### Input
| File | Format |
|------|--------|
| FASTA alignment | Standard FASTA (`.fasta` / `.fa`) |

### Processing
1. Parses the alignment with `Bio.AlignIO`.
2. Computes an identity-based distance matrix (`Bio.Phylo.TreeConstruction.DistanceCalculator`).
3. Builds a Neighbour-Joining tree (`DistanceTreeConstructor.nj`).
4. Saves the tree in Newick format.

### Output
A Newick (`.nwk`) file with `_tree` appended before the extension.

### Example
```bash
python -m phylocartoplot.preprocessing.build_phylogenetic_tree \
    sequences.fasta
# produces: sequences_tree.nwk
```

---

## Step 4 – Visualize Tree on Map (`tree_to_map_raster.py`)

**Purpose:** Render the phylogenetic tree as a geographic overlay on a raster (GeoTIFF) base map.

### Input
| File | Description |
|------|-------------|
| Newick tree (`.nwk`) | Phylogenetic tree from Step 3 |
| GPS/coordinates CSV | Columns: `node_name`, `longitude`, `latitude` |
| Offsets CSV | Per-node drawing offsets to avoid label overlap |
| Raster GeoTIFF (optional) | Environmental or topographic raster for the background map |

### Key Classes

#### `RasterMetadata`
Reads a GeoTIFF and extracts per-band metadata. Optionally enriched with a JSON sidecar file.

```python
from phylocartoplot.visualisation.tree_to_map_raster import RasterMetadata
meta = RasterMetadata("enviro.tif", metadata_file="env_metadata.json")
```

#### `PhyloCartoPlotter`
Main visualization class. Accepts all inputs and exposes `plot()` / `save()` methods.

```python
from phylocartoplot.visualisation.tree_to_map_raster import PhyloCartoPlotter

plotter = PhyloCartoPlotter(
    nwk_file="tree.nwk",
    gps_file="coords.csv",
    offset_file="offsets.csv",
    raster_file="enviro.tif",
    raster_band=1
)
plotter.plot()
plotter.save(output_dir="output")
```

### CLI Usage
```bash
python -m phylocartoplot.visualisation.tree_to_map_raster \
    --nwk tree.nwk \
    --gps coords.csv \
    --offset offsets.csv \
    --raster enviro.tif \
    --band 1 \
    --output output/
```

### Customisation Points
| Parameter | Description |
|-----------|-------------|
| `raster_band` | Which raster band to display as background |
| `offset_file` | Fine-tune leaf-node label positions |
| Colour mapping | Trait values are mapped to a configurable colour palette |

---

## Orchestrated Pipeline (`prepare_data.py`)

Runs Steps 1–3 in sequence from a single command:

```bash
python -m phylocartoplot.preprocessing.prepare_data \
    --data gbif_coffea_ex3.csv \
    --nodes node_names.csv \
    --meta no_caffeine_nodes_w_specimen.csv \
    --fasta sequences.fasta
```

---

## Common Errors & Troubleshooting

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `FileNotFoundError` on raster | Wrong path to `.tif` file | Use absolute paths or verify working directory |
| Empty merged DataFrame | Specimen IDs do not match node names | Check key extraction logic in `format_gbif_data.py` |
| `KeyError: 'trait_value'` | Metadata CSV column name differs | Rename column to `trait_value` |
| Tree branch lengths of 0 | Identical sequences in FASTA | Review alignment; NJ requires variation |
| Cartopy projection errors | Incompatible CRS | Ensure raster is in WGS-84 (EPSG:4326) |

---

## Data Flow Diagrams

Visual diagrams (DrawIO format) are available in `workflow/docs/`:
- `pipeline.drawio` – Full pipeline overview
- `coord_transformations.drawio` – Coordinate transformation details
- `Ex1_raster.drawio` – Use case 1 raster example
