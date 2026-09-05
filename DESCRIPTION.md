Overlay phylogenetic trees on geographic raster maps. Combines Newick-formatted trees with GBIF occurrence data and environmental layers (GeoTIFF). Supports arbitrary trait values. Works with any taxa, region, or raster dataset. Includes preprocessing for formatting coordinates, merging metadata, building trees, and visualization via Jupyter or CLI.

## Workflow

1. **Format GBIF data**: Clean and structure occurrence records with geographic coordinates
2. **Add metadata**: Merge trait or specimen-level values with coordinate data
3. **Build tree**: Construct phylogenetic trees from sequences (FASTA input)
4. **Visualize**: Render trees on raster backgrounds with value-to-color mapping

## I/O

- **Input**: Newick tree, CSV coordinates/traits, GeoTIFF raster, sequence alignment (FASTA)
- **Output**: SVG/PNG visualization, formatted CSV data

## Dependencies

Python ≥3.8, Biopython, Cartopy, Rasterio, Matplotlib, pandas, numpy, scikit-image

## Use Cases

Phylogeography, biogeographic trait distribution, spatial evolutionary patterns, comparative species analysis
