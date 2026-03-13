"""
PhyloCartoPlot: Phylogeographic visualization combining phylogenetic trees with raster/geographic data.

Creates side-by-side visualization with:
- Left panel: Phylogenetic tree colored by trait values
- Right panel: Raster layer or geographic map with occurrence points
- Connection lines linking tree nodes to locations

Usage:
    python tree_to_map_raster.py --nwk <tree.nwk> --gps <gps.csv> --offset <offsets.csv> [options]

Examples:
    # With raster (high contrast)
    python tree_to_map_raster.py \
        --nwk ../input/aligned_tree.nwk \
        --gps ../input/coords_w_metadata.csv \
        --offset ../input/offsets.csv \
        --raster ../input/enviro.tif \
        --raster-band 1
    
    # With raster and contrast enhancement
    python tree_to_map_raster.py \
        --nwk ../input/aligned_tree.nwk \
        --gps ../input/coords_w_metadata.csv \
        --offset ../input/offsets.csv \
        --raster ../input/enviro.tif \
        --raster-band 1 --raster-contrast histogram_eq
    
    # Custom extent and trait range
    python tree_to_map_raster.py \
        --nwk ../input/aligned_tree.nwk \
        --gps ../input/coords_w_metadata.csv \
        --offset ../input/offsets.csv \
        --extent 43 51 -27 -11 --vmin 0.01 --vmax 0.06 --trait-name "Trait (%)"
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
from Bio import Phylo
import argparse
import rasterio
from rasterio.plot import show
from pathlib import Path
from datetime import datetime
import json
from skimage import exposure


class RasterMetadata:
    """
    A class to extract and manage metadata from raster files.
    Similar to py_madaclim's MadaclimLayers, this class reads band information
    from GeoTIFF files and provides methods to query and retrieve band metadata.
    """

    def __init__(self, raster_path, metadata_file=None):
        """
        Initialize RasterMetadata by reading the raster file and extracting band information.

        Args:
            raster_path (str or Path): Path to the raster file
            metadata_file (str or Path, optional): Path to JSON metadata file with band descriptions
        """
        self.raster_path = Path(raster_path)
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.metadata = self._load_metadata() if self.metadata_file else {}
        self._validate_raster()
        self.bands_info = self._extract_bands_metadata()

    def _validate_raster(self):
        """Validate that the raster file exists and is readable."""
        if not self.raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {self.raster_path}")

        try:
            with rasterio.open(self.raster_path) as src:
                pass
        except Exception as e:
            raise IOError(f"Cannot open raster file: {e}")

    def _load_metadata(self) -> dict:
        """
        Load band metadata from a JSON file.

        Returns:
            dict: Metadata dictionary loaded from JSON file
        """
        if not self.metadata_file or not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata file: {e}")
            return {}

    def _extract_bands_metadata(self) -> pd.DataFrame:
        """
        Extract metadata for all bands in the raster file.
        Uses loaded metadata JSON file if available.

        Returns:
            pd.DataFrame: DataFrame containing band information with columns:
                - band_number: Band index (1-indexed)
                - band_name: Band description/name
                - band_description: Detailed description (if available)
        """
        # Define band mapping for madaclim environmental raster
        band_mapping = {
            1: {"name": "alt", "description": "Altitude (meters)"},
            2: {"name": "slo", "description": "Slope (degrees)"},
            3: {"name": "asp", "description": "Aspect; clockwise from North (degrees)"},
            4: {"name": "solrad", "description": "Solar radiation (Wh.m-2.day-1)"},
            5: {"name": "geo", "description": "Rock types (Geology)"},
            6: {"name": "soi", "description": "Soil types"},
            7: {"name": "veg", "description": "Vegetation types"},
            8: {"name": "wat", "description": "Watersheds"},
            9: {"name": "forcov", "description": "Forest cover (%)"},
        }

        # If metadata was loaded, use table_4 for non-categorical bands
        if self.metadata and "table_4" in self.metadata:
            table_4 = self.metadata["table_4"]
            layer_names = table_4.get("layer_name", [])
            layer_descs = table_4.get("layer_description", [])

            for i, (name, desc) in enumerate(zip(layer_names, layer_descs)):
                if (i + 1) in band_mapping:
                    band_mapping[i + 1] = {"name": name, "description": desc}

        bands_list = []

        with rasterio.open(self.raster_path) as src:
            print(f"\nRaster file: {self.raster_path.name}")
            print(f"Total bands: {src.count}")
            print(f"CRS: {src.crs}")
            print(f"Shape: {src.height} x {src.width}")
            print("-" * 80)

            for band_idx in range(1, src.count + 1):
                # Use mapping or fallback to generic name
                if band_idx in band_mapping:
                    band_name = band_mapping[band_idx]["name"]
                    band_description = band_mapping[band_idx]["description"]
                else:
                    band_name = f"Band {band_idx}"
                    band_description = f"Band {band_idx}"

                bands_list.append(
                    {
                        "band_number": band_idx,
                        "band_name": band_name,
                        "band_description": band_description,
                    }
                )

                print(f"Band {band_idx}: {band_name:8} | {band_description}")

        print("-" * 80)
        df = pd.DataFrame(bands_list)
        return df

    def get_band_name(self, band_number: int) -> str:
        """Get the short name of a specific band (e.g., 'alt', 'slo')."""
        if band_number < 1 or band_number > len(self.bands_info):
            raise ValueError(
                f"Band {band_number} is out of range (1-{len(self.bands_info)})"
            )
        return self.bands_info.iloc[band_number - 1]["band_name"]

    def get_band_description(self, band_number: int) -> str:
        """Get the full description of a specific band."""
        if band_number < 1 or band_number > len(self.bands_info):
            raise ValueError(
                f"Band {band_number} is out of range (1-{len(self.bands_info)})"
            )
        return self.bands_info.iloc[band_number - 1]["band_description"]

    def get_all_bands_info(self) -> pd.DataFrame:
        """Get complete metadata for all bands."""
        return self.bands_info.copy()

    def display_bands(self):
        """Print a formatted table of all bands and their information."""
        print("\n" + "=" * 80)
        print("RASTER BANDS METADATA")
        print("=" * 80)
        for _, row in self.bands_info.iterrows():
            print(
                f"Band {row['band_number']}: {row['band_name']:8} | {row['band_description']}"
            )
        print("=" * 80 + "\n")


def custom_label(clade):
    """Return node label for terminal clades, None for internal nodes."""
    if clade.is_terminal():
        return clade.name
    else:
        return None


def calc_node_positions(tree, x_start, x_end, y_start, y_step):
    """Calculate positions for all nodes in the tree."""
    if tree.is_terminal():
        x_pos = x_start
        y_pos = y_start
        y_start += y_step
    else:
        x_pos = (x_start + x_end) / 2
        y_pos = y_start

        child_y_start = y_start
        for child in tree.clades:
            child_x_pos, child_y_pos, y_start = calc_node_positions(
                child, x_start, x_end, y_start, y_step
            )
            x_start = child_x_pos

        y_pos = (y_start + child_y_start) / 2

    tree.position = (x_pos, y_pos)
    return x_pos, y_pos, y_start


def get_x_offset(node_name, offsets_dict):
    """Get x-axis offset for a node, default to 0 if not found."""
    return offsets_dict.get(node_name, 0)


def plot_adjusted_node(ax, node, y_offset, offsets_dict, gps):
    """Plot a single tree node with color from GPS data."""
    x, y = node.position
    x_offset = get_x_offset(node.name, offsets_dict)
    x += x_offset
    y += y_offset

    # Default color if node name is not found
    color = "grey"

    # Check if node name is in the gps DataFrame and set color accordingly
    if node.name in gps["specimen_id"].values:
        color = gps[gps["specimen_id"] == node.name]["color"].values[0]

    ax.plot(
        x,
        y,
        "o",
        markersize=8,
        markerfacecolor=color,
        markeredgewidth=2,
        markeredgecolor="black",
    )
    return x, y


def value_to_color(val, vmin, vmax):
    """
    Map trait values to colors using reversed viridis colormap.
    
    Args:
        val: Trait value to convert
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
    
    Returns:
        Hex color string
    """
    if val == 0.0:
        return "grey"
    elif vmin < val < vmax:
        # Use reversed viridis for better visual
        base = plt.get_cmap("viridis")
        newcolors = base(np.linspace(0, 0.8, 256))
        viridis_no_yellow = mcolors.ListedColormap(newcolors, name="viridis_no_yellow")
        cmap = viridis_no_yellow.reversed()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return mcolors.to_hex(cmap(norm(val)))
    else:
        return "black"


def enhance_raster_contrast(raster_data, method="percentile"):
    """
    Enhance raster contrast for better visibility.
    
    Args:
        raster_data: Input raster array
        method: 'percentile' (default), 'histogram_eq', or 'sigmoid'
    
    Returns:
        Normalized raster array (0-1 range)
    """
    if method == "percentile":
        # Stretch using percentiles (0.5-99.5)
        vmin = np.percentile(raster_data, 0.5)
        vmax = np.percentile(raster_data, 99.5)
        normalized = np.clip((raster_data - vmin) / (vmax - vmin), 0, 1)
        
    elif method == "histogram_eq":
        # Histogram equalization
        valid_mask = ~np.isnan(raster_data)
        normalized = raster_data.copy().astype(float)
        if np.any(valid_mask):
            data_valid = raster_data[valid_mask]
            data_eq = exposure.equalize_hist(data_valid)
            normalized[valid_mask] = data_eq
        
    elif method == "sigmoid":
        # Sigmoid contrast stretch
        vmin = np.percentile(raster_data, 0.5)
        vmax = np.percentile(raster_data, 99.5)
        normalized = (raster_data - vmin) / (vmax - vmin)
        normalized = 1 / (1 + np.exp(-15 * (normalized - 0.5)))
    else:
        raise ValueError(f"Unknown contrast method: {method}")
    
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="PhyloCartoPlot: Create phylogeographic visualizations with raster layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With raster (high contrast)
  python tree_to_map_raster.py --nwk tree.nwk --gps gps.csv --offset offsets.csv \\
      --raster enviro.tif --raster-band 1
  
  # With raster and contrast enhancement
  python tree_to_map_raster.py --nwk tree.nwk --gps gps.csv --offset offsets.csv \\
      --raster enviro.tif --raster-band 1 --raster-contrast histogram_eq
  
  # Custom extent and trait range
  python tree_to_map_raster.py --nwk tree.nwk --gps gps.csv --offset offsets.csv \\
      --extent 43 51 -27 -11 --vmin 0.01 --vmax 0.06 --trait-name "Trait (%)"
        """
    )

    # Required arguments
    parser.add_argument(
        "--nwk", type=str, required=True,
        help="Path to phylogenetic tree (Newick format)"
    )
    parser.add_argument(
        "--gps", type=str, required=True,
        help="Path to GPS coordinates with trait values (requires trait_value column)"
    )
    parser.add_argument(
        "--offset", type=str, required=True,
        help="Path to node position adjustments (XOffset column)"
    )

    # Optional raster arguments
    parser.add_argument(
        "--raster", type=str, default=None,
        help="Path to raster file (GeoTIFF)"
    )
    parser.add_argument(
        "--raster-band", type=int, default=1,
        help="Band number in raster file (default: 1)"
    )
    parser.add_argument(
        "--raster-metadata", type=str, default=None,
        help="Path to raster metadata JSON file"
    )
    parser.add_argument(
        "--raster-contrast", type=str, default="percentile",
        choices=["percentile", "histogram_eq", "sigmoid"],
        help="Contrast enhancement method (default: percentile)"
    )
    parser.add_argument(
        "--raster-cmap", type=str, default="terrain",
        help="Colormap for raster (default: terrain). Try: gray, viridis, hot, cool"
    )

    # Optional map arguments
    parser.add_argument(
        "--extent", type=float, nargs=4,
        default=[43, 51, -27, -11],
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
        help="Map extent [west, east, south, north] (default: Madagascar)"
    )

    # Optional color arguments
    parser.add_argument(
        "--vmin", type=float, default=None,
        help="Trait colormap minimum (auto-calculated if not provided)"
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Trait colormap maximum (auto-calculated if not provided)"
    )

    # Optional label arguments
    parser.add_argument(
        "--trait-name", type=str, default="Trait Value",
        help="Name of the trait for labels (default: Trait Value)"
    )

    # Optional output arguments
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: ../output/ relative to GPS file)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PhyloCartoPlot: Phylogeographic Visualization with Raster")
    print("=" * 80)
    print(f"\nLoading data...")
    print(f"  Tree:    {args.nwk}")
    print(f"  GPS:     {args.gps}")
    print(f"  Offsets: {args.offset}")

    # Load phylogenetic tree
    tree = Phylo.read(args.nwk, "newick")
    print(f"  ✓ Loaded tree with {len(list(tree.get_terminals()))} species")

    # Load GPS data
    gps = pd.read_csv(args.gps)
    print(f"  ✓ Loaded {len(gps)} GPS records")

    # Load node offsets
    offsets_df = pd.read_csv(args.offset)
    offsets_dict = pd.Series(
        offsets_df.XOffset.values, index=offsets_df.NodeName
    ).to_dict()
    print(f"  ✓ Loaded {len(offsets_dict)} node offsets")

    # Load raster if provided
    raster_data = None
    band_description = None
    if args.raster:
        try:
            raster_metadata = RasterMetadata(args.raster, metadata_file=args.raster_metadata)
            raster_metadata.display_bands()
            
            band_description = raster_metadata.get_band_description(args.raster_band)
            print(f"Selected band: {args.raster_band}")
            print(f"Band label: {band_description}\n")

            with rasterio.open(args.raster) as src:
                raster_data = src.read(args.raster_band)
            
            print(f"Raster data shape: {raster_data.shape}")
            print(f"Raster data type: {raster_data.dtype}")
            print(f"✓ Loaded raster: {args.raster} (band {args.raster_band})")
            
        except Exception as e:
            print(f"  ✗ Error loading raster: {e}")
            raster_data = None

    # Auto-calculate trait value range if not provided
    if args.vmin is None:
        args.vmin = gps[gps["trait_value"] > 0]["trait_value"].quantile(0.05)
    if args.vmax is None:
        args.vmax = gps[gps["trait_value"] > 0]["trait_value"].quantile(0.95)

    print(f"\nConfiguration:")
    if raster_data is not None:
        print(f"  Display mode:       Raster layer")
        print(f"  Raster:             {args.raster}")
        print(f"  Contrast method:    {args.raster_contrast}")
    else:
        print(f"  Display mode:       Geographic map")
        print(f"  Map extent:         {args.extent}")
    print(f"  Trait name:         {args.trait_name}")
    print(f"  Trait value range:  {args.vmin:.6f} - {args.vmax:.6f}")

    # Create figure
    print(f"\nCreating visualization...")
    fig = plt.figure(figsize=(35, 12))

    # ============================
    # PHYLOGENETIC TREE (LEFT)
    # ============================

    y_step = 1
    calc_node_positions(tree.root, 0, 1, 0, y_step)

    ax_tree = fig.add_subplot(121)

    # Color nodes by trait value
    gps["color"] = gps["trait_value"].apply(lambda x: value_to_color(x, args.vmin, args.vmax))
    
    # Build text colors dictionary
    valmap = dict(zip(gps["specimen_id"], gps["trait_value"]))
    base = plt.get_cmap("viridis")
    newcolors = base(np.linspace(0, 0.8, 256))
    viridis_no_yellow = mcolors.ListedColormap(newcolors, name="viridis_no_yellow")
    cmap_rev = viridis_no_yellow.reversed()
    norm = mcolors.Normalize(vmin=args.vmin, vmax=args.vmax)

    text_colors = {
        raw: (
            "grey" if valmap[raw] == 0.0
            else (
                mcolors.to_hex(cmap_rev(norm(valmap[raw])))
                if args.vmin < valmap[raw] < args.vmax
                else "black"
            )
        )
        for raw in gps["specimen_id"]
    }

    # Plot tree
    Phylo.draw(
        tree,
        do_show=False,
        axes=ax_tree,
        label_func=custom_label,
        label_colors=text_colors,
    )

    for txt in ax_tree.texts:
        txt.set_fontsize(24)
        txt.set_fontstyle("italic")

    ax_tree.set_frame_on(False)
    ax_tree.axis("off")
    ax_tree.set_xlim(-0.05, 1)
    ax_tree.set_ylim(0, max(node.position[1] for node in tree.get_terminals()) + 2)

    print(f"  ✓ Tree plotted")

    # Store node positions for connection lines
    rows = []
    for clade in tree.find_clades():
        if clade.is_terminal():
            label = clade.name
            x, y = plot_adjusted_node(ax_tree, clade, y_step, offsets_dict, gps)
            rows.append([label, (x, y)])

    df = pd.DataFrame(rows, columns=["ID", "Coordinates"])

    # ============================
    # RIGHT PANEL (RASTER OR MAP)
    # ============================

    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    ax2.set_extent(args.extent)
    ax2.set_frame_on(False)
    ax2.axis("off")

    if raster_data is not None:
        # RASTER MODE
        print(f"  ✓ Enhancing raster contrast ({args.raster_contrast})...")
        
        # Apply contrast enhancement
        raster_enhanced = enhance_raster_contrast(raster_data, method=args.raster_contrast)
        
        # Mask negative values (water in DEM)
        raster_masked = raster_enhanced.copy().astype(float)
        raster_masked[raster_data < 0] = np.nan

        # Display raster
        im = ax2.imshow(
            raster_masked,
            transform=ccrs.PlateCarree(),
            cmap=args.raster_cmap,
            extent=args.extent,
            origin="upper",
            alpha=0.95,
            vmin=0,
            vmax=1,
        )

        # Add borders on top
        ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")

        # Add raster colorbar on right
        cbar_raster = fig.colorbar(
            im,
            ax=ax2,
            orientation="vertical",
            pad=0.02,
            location="right",
            shrink=0.9,
            aspect=30,
            label=band_description,
        )
        cbar_raster.set_label(band_description, fontsize=20)
        cbar_raster.ax.tick_params(labelsize=18)

        print(f"  ✓ Raster plotted with contrast enhancement")

    else:
        # MAP MODE
        ax2.coastlines(resolution="10m")
        ax2.add_feature(cfeature.LAND)
        ax2.add_feature(cfeature.OCEAN)
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.BORDERS)

        print(f"  ✓ Map plotted")

    # Plot GPS points on top
    for _, row in gps.iterrows():
        is_zero = row["trait_value"] == 0.0
        ax2.plot(
            row["longitude"],
            row["latitude"],
            "o",
            markersize=8,
            markerfacecolor=row["color"],
            markeredgewidth=2,
            markeredgecolor="black",
            alpha=0.2 if is_zero else 1.0,
        )

    # Add trait colorbar on left
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_rev)
    cbar = fig.colorbar(
        sm,
        ax=ax_tree,
        orientation="vertical",
        pad=0.02,
        location="left",
        shrink=0.9,
        aspect=30,
    )
    cbar.set_label(args.trait_name, fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    # ============================
    # CONNECTION LINES
    # ============================

    gps_grouped = (
        gps.groupby("specimen_id")[["longitude", "latitude", "color"]]
        .apply(lambda x: list(zip(x["longitude"], x["latitude"], x["color"])))
        .to_dict()
    )
    trait_dict = dict(zip(gps["specimen_id"], gps["trait_value"]))

    for index, row in df.iterrows():
        specimen = row["ID"]
        if specimen in gps_grouped:
            for longitude, latitude, color in gps_grouped[specimen]:
                trait_val = trait_dict.get(specimen, 0.0)
                lw = 2.0 if trait_val > 0 else 0.8
                line_alpha = 0.2 if color == "grey" else 0.2

                con = ConnectionPatch(
                    xyA=row["Coordinates"],
                    coordsA="data",
                    xyB=(longitude, latitude),
                    coordsB="data",
                    axesA=ax_tree,
                    axesB=ax2,
                    color=color,
                    linewidth=lw,
                    linestyle="--",
                    alpha=line_alpha,
                    zorder=2,
                )
                fig.add_artist(con)

    print(f"  ✓ Connection lines drawn")

    # ============================
    # SAVE OUTPUT
    # ============================

    fig.subplots_adjust(wspace=1.5)
    plt.tight_layout()

    # Determine output directory
    output_dir = Path(args.output) if args.output else Path(args.gps).parent.parent / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tree_basename = Path(args.nwk).stem
    suffix = "_raster" if raster_data is not None else "_map"
    svg_file = output_dir / f"{tree_basename}_to{suffix}_{timestamp}.svg"
    png_file = output_dir / f"{tree_basename}_to{suffix}_{timestamp}.png"

    # Save figures
    plt.savefig(str(svg_file), format="svg", transparent=True, bbox_inches="tight", pad_inches=0.1, dpi=300)
    print(f"\n✓ Saved SVG: {svg_file}")

    plt.savefig(str(png_file), format="png", transparent=True, dpi=300)
    print(f"✓ Saved PNG: {png_file}")

    plt.close(fig)

    print(f"\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()