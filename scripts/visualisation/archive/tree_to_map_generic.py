"""
PhyloCartoPlot: Phylogeographic visualization combining phylogenetic trees with geographic maps.

Creates side-by-side visualization with:
- Left panel: Phylogenetic tree colored by trait values
- Right panel: Geographic map with occurrence points
- Connection lines linking tree nodes to geographic locations

Usage:
    python tree_to_map.py --nwk <tree.nwk> --gps <gps.csv> --offset <offsets.csv> [options]

Example:
    python tree_to_map.py \
        --nwk ../input/aligned_tree.nwk \
        --gps ../input/coords_w_metadata.csv \
        --offset ../input/offsets.csv \
        --extent 43 51 -27 -11 \
        --trait-name "Trait Value"
"""

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
from Bio import Phylo
from pathlib import Path
from datetime import datetime
import argparse


def custom_label(clade):
    """Return node label for terminal clades, None for internal nodes."""
    if clade.is_terminal():
        return clade.name
    else:
        return None


def calc_node_positions(tree, x_start, x_end, y_start, y_step):
    """
    Calculate positions for all nodes in the tree.
    
    Args:
        tree: Phylogenetic tree (Bio.Phylo.BaseTree.Tree)
        x_start: Starting x position
        x_end: Ending x position
        y_start: Starting y position
        y_step: Vertical step between terminal nodes
    
    Returns:
        Tuple of (x_pos, y_pos, y_start) for this node
    """
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
    """
    Plot a single tree node with color from GPS data.
    
    Args:
        ax: Matplotlib axes
        node: Phylogenetic node (Bio.Phylo.BaseTree.Clade)
        y_offset: Y-axis offset
        offsets_dict: Dictionary of x offsets by node name
        gps: DataFrame with specimen_id and color columns
    
    Returns:
        Tuple of (x, y) position
    """
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
    Map trait values to colors using viridis colormap.
    
    Args:
        val: Trait value to convert
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
    
    Returns:
        Hex color string
    """
    if val == 0.00:
        # No data
        return "grey"
    elif vmin < val < vmax:
        # Within range: use viridis colormap
        base_cmap = plt.get_cmap("viridis")
        newcolors = base_cmap(np.linspace(0, 0.8, 256))
        viridis_no_yellow = mcolors.ListedColormap(newcolors, name="viridis_no_yellow")
        cmap = viridis_no_yellow
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return mcolors.to_hex(cmap(norm(val)))
    else:
        # Outside range
        return "black"


def main():
    """Main function for PhyloCartoPlot visualization."""
    
    parser = argparse.ArgumentParser(
        description="PhyloCartoPlot: Create phylogeographic visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (Madagascar extent, auto color range)
  python tree_to_map.py --nwk tree.nwk --gps gps.csv --offset offsets.csv
  
  # Custom map extent and trait name
  python tree_to_map.py --nwk tree.nwk --gps gps.csv --offset offsets.csv \
      --extent -10 40 -35 5 --trait-name "Elevation (m)"
  
  # Specific color range
  python tree_to_map.py --nwk tree.nwk --gps gps.csv --offset offsets.csv \
      --vmin 100 --vmax 2000 --trait-name "Elevation (m)"
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--nwk", type=str, required=True, 
        help="Path to phylogenetic tree (Newick format)"
    )
    parser.add_argument(
        "--gps", type=str, required=True,
        help="Path to GPS coordinates with trait values (must have trait_value column)"
    )
    parser.add_argument(
        "--offset", type=str, required=True,
        help="Path to node position adjustments (XOffset column)"
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
        help="Colormap minimum value (auto-calculated from data if not provided)"
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Colormap maximum value (auto-calculated from data if not provided)"
    )
    
    # Optional label arguments
    parser.add_argument(
        "--trait-name", type=str, default="Trait Value",
        help="Name of the trait for display labels (default: Trait Value)"
    )
    
    # Optional output arguments
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: ../output/ relative to GPS file)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PhyloCartoPlot: Phylogeographic Visualization")
    print("=" * 70)
    print(f"\nLoading data...")
    print(f"  Tree:    {args.nwk}")
    print(f"  GPS:     {args.gps}")
    print(f"  Offsets: {args.offset}")

    # Load input files
    tree = Phylo.read(args.nwk, "newick")
    print(f"  ✓ Loaded tree with {len(list(tree.get_terminals()))} species")

    gps = pd.read_csv(args.gps)
    print(f"  ✓ Loaded {len(gps)} GPS records")

    offsets_df = pd.read_csv(args.offset)
    offsets_dict = pd.Series(
        offsets_df.XOffset.values, index=offsets_df.NodeName
    ).to_dict()
    print(f"  ✓ Loaded {len(offsets_dict)} node offsets")

    # Auto-calculate color range if not provided
    if args.vmin is None:
        args.vmin = gps[gps["trait_value"] > 0]["trait_value"].quantile(0.05)
    if args.vmax is None:
        args.vmax = gps[gps["trait_value"] > 0]["trait_value"].quantile(0.95)

    
    print(f"\nConfiguration:")
    print(f"  Map extent:        {args.extent}")
    print(f"  Trait name:        {args.trait_name}")
    print(f"  Trait value range: {args.vmin:.6f} - {args.vmax:.6f}")

    # Create figure
    print(f"\nCreating visualization...")
    fig = plt.figure(figsize=(26, 11))

    # ============================
    # PHYLOGENETIC TREE (LEFT)
    # ============================

    # Calculate tree node positions
    y_step = 1
    calc_node_positions(tree.root, 0, 1, 0, y_step)

    ax_tree = fig.add_subplot(121)

    # Color nodes by trait value
    gps["color"] = gps["trait_value"].apply(lambda x: value_to_color(x, args.vmin, args.vmax))
    raw_color_map = dict(zip(gps["specimen_id"], gps["color"]))
    text_colors = {raw: col for raw, col in raw_color_map.items()}

    # Plot tree
    Phylo.draw(
        tree,
        do_show=False,
        axes=ax_tree,
        label_func=custom_label,
        label_colors=text_colors,
    )

    # Format tree labels
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
            x, y = plot_adjusted_node(
                ax_tree, clade, y_step, offsets_dict, gps
            )
            rows.append([label, (x, y)])

    df = pd.DataFrame(rows, columns=["ID", "Coordinates"])

    # ============================
    # GEOGRAPHIC MAP (RIGHT)
    # ============================

    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    ax2.set_extent(args.extent)

    # Plot GPS points on map
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

    # Add map features
    ax2.coastlines(resolution="10m")
    ax2.add_feature(cfeature.LAND)
    # ax2.add_feature(cfeature.OCEAN)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.set_frame_on(False)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    print(f"  ✓ Map plotted")

    # Add legend
    legend_handles = [
        mlines.Line2D([], [], color="grey", marker="o", linestyle="None",
                      label="No data (0)"),
        mlines.Line2D([], [], color="purple", marker="o", linestyle="None",
                      label=f"Range ({args.vmin:.4f}–{args.vmax:.4f})"),
        mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                      label="Outside range"),
    ]
    ax2.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=12)

    # Add colorbar
    base_cmap = plt.get_cmap("viridis")
    newcolors = base_cmap(np.linspace(0, 0.8, 256))
    viridis_no_yellow = mcolors.ListedColormap(newcolors, name="viridis_no_yellow")
    cmap = viridis_no_yellow

    norm = mpl.colors.Normalize(vmin=args.vmin, vmax=args.vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    cbar = fig.colorbar(sm, ax=ax2, orientation="vertical", pad=0.02)
    cbar.set_label(args.trait_name, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # ============================
    # CONNECTION LINES
    # ============================

    # Group GPS data by specimen ID
    gps_grouped = (
        gps.groupby("specimen_id")[["longitude", "latitude", "color"]]
        .apply(lambda x: list(zip(x["longitude"], x["latitude"], x["color"])))
        .to_dict()
    )
    trait_dict = dict(zip(gps["specimen_id"], gps["trait_value"]))

    # Draw connection lines between tree and map
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

    plt.tight_layout()

    # Determine output directory
    output_dir = Path(args.output) if args.output else Path(args.gps).parent.parent / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tree_basename = Path(args.nwk).stem
    svg_file = output_dir / f"{tree_basename}_to_map_{timestamp}.svg"
    png_file = output_dir / f"{tree_basename}_to_map_{timestamp}.png"

    # Save figures
    plt.savefig(str(svg_file), format="svg", dpi=300)
    print(f"\n✓ Saved SVG: {svg_file}")

    plt.savefig(str(png_file), format="png", dpi=300)
    print(f"✓ Saved PNG: {png_file}")

    plt.close(fig)

    print(f"\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()