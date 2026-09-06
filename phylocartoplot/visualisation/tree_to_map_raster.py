"""
PhyloGeoPlot: Phylogeographic visualization combining phylogenetic trees with raster/geographic data.

Can be used as a CLI tool or imported as classes in Jupyter notebooks.

CLI Usage:
    python tree_to_map_raster.py --nwk <tree.nwk> --gps <gps.csv> --offset <offsets.csv> [options]

Jupyter Usage:
    from tree_to_map_raster import PhyloGeoPlotter

    plotter = PhyloGeoPlotter(
        nwk_file="tree.nwk",
        gps_file="gps.csv",
        offset_file="offsets.csv",
        raster_file="enviro.tif",
        raster_band=1
    )
    plotter.plot()
    plotter.save(output_dir="output")
"""
