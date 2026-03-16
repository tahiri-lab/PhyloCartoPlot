"""
PhyloCartoPlot: Phylogeographic visualization combining phylogenetic trees with raster/geographic data.

Can be used as a CLI tool or imported as classes in Jupyter notebooks.

CLI Usage:
    python tree_to_map_raster.py --nwk <tree.nwk> --gps <gps.csv> --offset <offsets.csv> [options]

Jupyter Usage:
    from tree_to_map_raster import PhyloCartoPlotter

    plotter = PhyloCartoPlotter(
        nwk_file="tree.nwk",
        gps_file="gps.csv",
        offset_file="offsets.csv",
        raster_file="enviro.tif",
        raster_band=1
    )
    plotter.plot()
    plotter.save(output_dir="output")
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

    def __init__(self, raster_path, metadata_file=None, verbose=True):
        """
        Initialize RasterMetadata by reading the raster file and extracting band information.

        Args:
            raster_path (str or Path): Path to the raster file
            metadata_file (str or Path, optional): Path to JSON metadata file with band descriptions
            verbose (bool): Print band information (default: True)
        """
        self.raster_path = Path(raster_path)
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.metadata = self._load_metadata() if self.metadata_file else {}
        self.verbose = verbose
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
        """Load band metadata from a JSON file."""
        if not self.metadata_file or not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load metadata file: {e}")
            return {}

    def _extract_bands_metadata(self) -> pd.DataFrame:
        """Extract metadata for all bands in the raster file."""
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

        if self.metadata and "table_4" in self.metadata:
            table_4 = self.metadata["table_4"]
            layer_names = table_4.get("layer_name", [])
            layer_descs = table_4.get("layer_description", [])

            for i, (name, desc) in enumerate(zip(layer_names, layer_descs)):
                if (i + 1) in band_mapping:
                    band_mapping[i + 1] = {"name": name, "description": desc}

        bands_list = []

        with rasterio.open(self.raster_path) as src:
            if self.verbose:
                print(f"\nRaster file: {self.raster_path.name}")
                print(f"Total bands: {src.count}")
                print(f"CRS: {src.crs}")
                print(f"Shape: {src.height} x {src.width}")
                print("-" * 80)

            for band_idx in range(1, src.count + 1):
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

                if self.verbose:
                    print(f"Band {band_idx}: {band_name:8} | {band_description}")

        if self.verbose:
            print("-" * 80)

        df = pd.DataFrame(bands_list)
        return df

    def get_band_name(self, band_number: int) -> str:
        """Get the short name of a specific band."""
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


class PhyloCartoPlotter:
    """
    Main class for creating phylogeographic visualizations.

    Combines phylogenetic trees, geographic maps or raster layers, and GPS coordinates.
    """

    def __init__(
        self,
        nwk_file,
        gps_file,
        offset_file,
        raster_file=None,
        raster_band=1,
        raster_metadata_file=None,
        raster_contrast="percentile",
        raster_cmap="terrain",
        extent=[43, 51, -27, -11],
        vmin=None,
        vmax=None,
        trait_name="Trait Value",
        color_map=None,
        color_to_label=None,
        legend_config=None,
        verbose=True,
    ):
        """
        Initialize PhyloCartoPlotter.

        Args:
            nwk_file: Path to phylogenetic tree (Newick format)
            gps_file: Path to GPS coordinates CSV (must have trait_value column)
            offset_file: Path to node position adjustments CSV
            raster_file: Path to raster file (GeoTIFF), optional
            raster_band: Band number in raster file (default: 1)
            raster_metadata_file: Path to raster metadata JSON file
            raster_contrast: Contrast enhancement method ('percentile', 'histogram_eq', 'sigmoid')
            raster_cmap: Colormap for raster (default: 'terrain')
            extent: Map extent [west, east, south, north]
            vmin: Trait value colormap minimum (auto-calculated if None)
            vmax: Trait value colormap maximum (auto-calculated if None)
            trait_name: Name of the trait for labels
            color_map: Dict mapping numeric trait_value to colors (e.g., {1: "red", 2: "blue"}).
                Required if legend_config['show_legend']=True. If None, uses continuous colormap.
            color_to_label: Dict mapping colors to categorical labels 
                (e.g., {"red": "Bassin d'Islande", "blue": "Bassin d'Irminger"}).
                Required if legend_config['show_legend']=True.
            legend_config: Dict with legend customizations. If provided with show_legend=True,
                displays categorical legend and hides colorbar. Valid keys:
                - 'show_legend': Enable legend (bool, default: False)
                - 'loc': Position ('lower right', 'upper left', etc.)
                - 'fontsize': Label font size (int, default: 10)
                - 'title_fontsize': Title font size (int, default: 12)
                - 'title': Legend title (str or None, default: None)
                - 'frameon': Show border (bool, default: True)
                - 'markerscale': Marker scaling factor (float, default: 1.2)
                - 'marker_size': Circle size (int, default: 8)
                - 'marker_edge_color': Outline color (str, default: 'black')
                - 'framealpha': Background transparency (0-1, default: 0.9)
                - 'edgecolor': Border color (str, default: 'black')
                - 'fancybox': Rounded corners (bool, default: True)
                - 'shadow': Drop shadow (bool, default: False)
            verbose: Print progress messages (default: True)
        """
        self.nwk_file = nwk_file
        self.gps_file = gps_file
        self.offset_file = offset_file
        self.raster_file = raster_file
        self.raster_band = raster_band
        self.raster_metadata_file = raster_metadata_file
        self.raster_contrast = raster_contrast
        self.raster_cmap = raster_cmap
        self.extent = extent
        self.vmin = vmin
        self.vmax = vmax
        self.trait_name = trait_name
        self.verbose = verbose

        # ===== COLOR MAPPINGS FOR CATEGORICAL LEGEND =====
        self.color_map = color_map or {}  # {numeric_value: color}
        self.color_to_label = color_to_label or {}  # {color: label}
        # ===================================================

        # ===== LEGEND CONFIGURATION =====
        self.legend_config = {
            'show_legend': False,              # Legend disabled by default
            'loc': 'lower left',               # Position
            'bbox_to_anchor': (0.01, 0.01),   # Normalized coords: (x, y) for lower left
            'fontsize': 10,                    # Label font size
            'title_fontsize': 12,              # Title font size
            'title': None,                     # Legend title (None = no title)
            'frameon': True,                   # Border around legend
            'markerscale': 1.2,                # Marker size scaling
            'marker_size': 8,                  # Circle marker size
            'marker_edge_color': 'black',      # Outline color
            'framealpha': 0.9,                 # Legend background transparency
            'edgecolor': 'black',              # Legend border color
            'fancybox': True,                  # Rounded corners
            'shadow': False,                   # Drop shadow
        }
        
        # Override defaults with user-provided config
        if legend_config is not None:
            self.legend_config.update(legend_config)
        # ================================

        # Data containers
        self.tree = None
        self.gps = None
        self.offsets_dict = None
        self.raster_data = None
        self.band_description = None
        self.fig = None
        self.ax_tree = None
        self.ax2 = None

        # Load data
        self._load_data()
        self._prepare_colors()

    def _load_data(self):
        """Load all input data."""
        if self.verbose:
            print("=" * 80)
            print("PhyloCartoPlot: Loading Data")
            print("=" * 80)
            print(f"\nLoading data...")
            print(f"  Tree:    {self.nwk_file}")
            print(f"  GPS:     {self.gps_file}")
            print(f"  Offsets: {self.offset_file}")

        # Load tree
        self.tree = Phylo.read(self.nwk_file, "newick")
        if self.verbose:
            print(
                f"  ✓ Loaded tree with {len(list(self.tree.get_terminals()))} species"
            )

        # Load GPS data
        self.gps = pd.read_csv(self.gps_file)
        if self.verbose:
            print(f"  ✓ Loaded {len(self.gps)} GPS records")

        # Apply color mapping if provided
        if self.color_map:
            self.gps['color'] = self.gps['trait_value'].map(self.color_map)
            if self.gps['color'].isna().any():
                missing_values = self.gps[self.gps['color'].isna()]['trait_value'].unique()
                if self.verbose:
                    print(f"  ⚠ Warning: Some trait_values not in color_map: {missing_values}")
            if self.verbose:
                print(f"  ✓ Applied color mapping to GPS records")

        # Load offsets
        offsets_df = pd.read_csv(self.offset_file)
        self.offsets_dict = pd.Series(
            offsets_df.XOffset.values, index=offsets_df.NodeName
        ).to_dict()
        if self.verbose:
            print(f"  ✓ Loaded {len(self.offsets_dict)} node offsets")

        # Load raster if provided
        if self.raster_file:
            self._load_raster()

        # Auto-calculate trait range if not provided
        if self.vmin is None:
            self.vmin = self.gps[self.gps["trait_value"] > 0]["trait_value"].quantile(
                0.05
            )
        if self.vmax is None:
            self.vmax = self.gps[self.gps["trait_value"] > 0]["trait_value"].quantile(
                0.95
            )

        if self.verbose:
            print(f"\nConfiguration:")
            if self.raster_data is not None:
                print(f"  Display mode:       Raster layer")
                print(f"  Raster:             {self.raster_file}")
                print(f"  Contrast method:    {self.raster_contrast}")
            else:
                print(f"  Display mode:       Geographic map")
                print(f"  Map extent:         {self.extent}")
            print(f"  Trait name:         {self.trait_name}")
            print(f"  Trait value range:  {self.vmin:.6f} - {self.vmax:.6f}")

    def _load_raster(self):
        """Load raster data."""
        try:
            raster_metadata = RasterMetadata(
                self.raster_file,
                metadata_file=self.raster_metadata_file,
                verbose=self.verbose,
            )

            if self.verbose:
                raster_metadata.display_bands()

            self.band_description = raster_metadata.get_band_description(
                self.raster_band
            )
            if self.verbose:
                print(f"Selected band: {self.raster_band}")
                print(f"Band label: {self.band_description}\n")

            with rasterio.open(self.raster_file) as src:
                self.raster_data = src.read(self.raster_band)

            if self.verbose:
                print(f"Raster data shape: {self.raster_data.shape}")
                print(f"Raster data type: {self.raster_data.dtype}")
                print(f"✓ Loaded raster: {self.raster_file} (band {self.raster_band})")

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Error loading raster: {e}")
            self.raster_data = None

    def _prepare_colors(self):
        """Prepare color mappings for trait values."""
        self.gps["color"] = self.gps["trait_value"].apply(self._value_to_color)

    def _value_to_color(self, val):
        """Map trait values to colors."""
        if val == 0.0:
            return "grey"
        elif self.vmin < val < self.vmax:
            base = plt.get_cmap("viridis")
            newcolors = base(np.linspace(0, 0.8, 256))
            viridis_no_yellow = mcolors.ListedColormap(
                newcolors, name="viridis_no_yellow"
            )
            cmap = viridis_no_yellow.reversed()
            norm = mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)
            return mcolors.to_hex(cmap(norm(val)))
        else:
            return "black"

    def _enhance_raster_contrast(self, raster_data, method="percentile"):
        """Enhance raster contrast for better visibility."""
        if method == "percentile":
            vmin = np.percentile(raster_data, 0.5)
            vmax = np.percentile(raster_data, 99.5)
            normalized = np.clip((raster_data - vmin) / (vmax - vmin), 0, 1)

        elif method == "histogram_eq":
            valid_mask = ~np.isnan(raster_data)
            normalized = raster_data.copy().astype(float)
            if np.any(valid_mask):
                data_valid = raster_data[valid_mask]
                data_eq = exposure.equalize_hist(data_valid)
                normalized[valid_mask] = data_eq

        elif method == "sigmoid":
            vmin = np.percentile(raster_data, 0.5)
            vmax = np.percentile(raster_data, 99.5)
            normalized = (raster_data - vmin) / (vmax - vmin)
            normalized = 1 / (1 + np.exp(-15 * (normalized - 0.5)))
        else:
            raise ValueError(f"Unknown contrast method: {method}")

        return normalized

    def _calc_node_positions(self, tree, x_start, x_end, y_start, y_step):
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
                child_x_pos, child_y_pos, y_start = self._calc_node_positions(
                    child, x_start, x_end, y_start, y_step
                )
                x_start = child_x_pos

            y_pos = (y_start + child_y_start) / 2

        tree.position = (x_pos, y_pos)
        return x_pos, y_pos, y_start

    def _custom_label(self, clade):
        """Return node label for terminal clades."""
        if clade.is_terminal():
            return clade.name
        else:
            return None

    def _get_x_offset(self, node_name):
        """Get x-axis offset for a node."""
        return self.offsets_dict.get(node_name, 0)
    
    def _get_color_from_trait(self, trait_value):
        """Get color from trait_value using color_map.
        
        If color_map is provided, use it. Otherwise return grey.
        """
        if self.color_map is None:
            return "grey"
        
        # Convert trait_value to int if it's a float ending in .0
        if isinstance(trait_value, float) and trait_value == int(trait_value):
            trait_value = int(trait_value)
        
        return self.color_map.get(trait_value, "grey")

    def _plot_adjusted_node(self, ax, node, y_offset):
        """Plot a single tree node with color from color_map."""
        x, y = node.position
        x_offset = self._get_x_offset(node.name)
        x += x_offset
        y += y_offset

        # Get color from color_map based on trait_value
        color = "grey"
        if node.name in self.gps["specimen_id"].values:
            trait_value = self.gps[self.gps["specimen_id"] == node.name]["trait_value"].values[0]
            color = self._get_color_from_trait(trait_value)

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

    def plot(self, figsize=(35, 12)):
        """Create the phylogeographic visualization."""
        if self.verbose:
            print(f"\nCreating visualization...")

        self.fig = plt.figure(figsize=figsize)

        # ============================
        # PHYLOGENETIC TREE (LEFT)
        # ============================

        y_step = 1
        self._calc_node_positions(self.tree.root, 0, 1, 0, y_step)

        self.ax_tree = self.fig.add_subplot(121)

        # Build text colors dictionary
        valmap = dict(zip(self.gps["specimen_id"], self.gps["trait_value"]))
        
        # Use categorical colors if color_map is provided and legend is enabled
        if self.color_map and self.legend_config.get('show_legend', False):
            # Categorical colors from color_map
            text_colors = {
                specimen_id: (
                    "grey"
                    if valmap[specimen_id] == 0.0
                    else self._get_color_from_trait(valmap[specimen_id])
                )
                for specimen_id in self.gps["specimen_id"]
            }
        else:
            # Continuous colors from colormap (default behavior)
            base = plt.get_cmap("viridis")
            newcolors = base(np.linspace(0, 0.8, 256))
            viridis_no_yellow = mcolors.ListedColormap(newcolors, name="viridis_no_yellow")
            cmap_rev = viridis_no_yellow.reversed()
            norm = mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)

            text_colors = {
                raw: (
                    "grey"
                    if valmap[raw] == 0.0
                    else (
                        mcolors.to_hex(cmap_rev(norm(valmap[raw])))
                        if self.vmin < valmap[raw] < self.vmax
                        else "black"
                    )
                )
                for raw in self.gps["specimen_id"]
            }

        # Plot tree
        Phylo.draw(
            self.tree,
            do_show=False,
            axes=self.ax_tree,
            label_func=self._custom_label,
            label_colors=text_colors,
        )

        for txt in self.ax_tree.texts:
            txt.set_fontsize(14)  # Increased from 12 to 14 for better visibility in raster
            txt.set_fontstyle("italic")

        self.ax_tree.set_frame_on(False)
        self.ax_tree.axis("off")
        self.ax_tree.set_xlim(-0.05, 1)
        self.ax_tree.set_ylim(
            0, max(node.position[1] for node in self.tree.get_terminals()) + 2
        )

        if self.verbose:
            print(f"  ✓ Tree plotted")

        # Store node positions
        rows = []
        for clade in self.tree.find_clades():
            if clade.is_terminal():
                label = clade.name
                x, y = self._plot_adjusted_node(self.ax_tree, clade, y_step)
                rows.append([label, (x, y)])

        df = pd.DataFrame(rows, columns=["ID", "Coordinates"])

        # ============================
        # RIGHT PANEL (RASTER OR MAP)
        # ============================

        self.ax2 = self.fig.add_subplot(122, projection=ccrs.PlateCarree())
        self.ax2.set_extent(self.extent)
        self.ax2.set_frame_on(False)
        self.ax2.axis("off")

        if self.raster_data is not None:
            # RASTER MODE
            if self.verbose:
                print(f"  ✓ Enhancing raster contrast ({self.raster_contrast})...")

            raster_enhanced = self._enhance_raster_contrast(
                self.raster_data, method=self.raster_contrast
            )
            raster_masked = raster_enhanced.copy().astype(float)
            raster_masked[self.raster_data < 0] = np.nan

            im = self.ax2.imshow(
                raster_masked,
                transform=ccrs.PlateCarree(),
                cmap=self.raster_cmap,
                extent=self.extent,
                origin="upper",
                alpha=0.75,
                vmin=0,
                vmax=1,
            )

            self.ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")

            cbar_raster = self.fig.colorbar(
                im,
                ax=self.ax2,
                orientation="vertical",
                pad=0.02,
                location="right",
                shrink=0.9,
                aspect=30,
                label=self.band_description,
            )
            cbar_raster.set_label(self.band_description, fontsize=20)
            cbar_raster.ax.tick_params(labelsize=18)

            if self.verbose:
                print(f"  ✓ Raster plotted with contrast enhancement")

        else:
            # MAP MODE
            self.ax2.coastlines(resolution="10m")
            self.ax2.add_feature(cfeature.LAND)
            # self.ax2.add_feature(cfeature.OCEAN)
            self.ax2.add_feature(cfeature.COASTLINE)
            self.ax2.add_feature(cfeature.BORDERS)
            self.ax2.set_frame_on(False)

            if self.verbose:
                print(f"  ✓ Map plotted")

        # Build colormap for continuous gradient (used when not using categorical legend)
        base = plt.get_cmap("viridis")
        newcolors = base(np.linspace(0, 0.8, 256))
        viridis_no_yellow = mcolors.ListedColormap(newcolors, name="viridis_no_yellow")
        cmap_rev = viridis_no_yellow.reversed()
        norm = mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)

        # Plot GPS points
        for _, row in self.gps.iterrows():
            is_zero = row["trait_value"] == 0.0
            
            # Use categorical colors if legend is enabled, otherwise use gradient
            if self.color_map and self.legend_config.get('show_legend', False):
                color = self._get_color_from_trait(row["trait_value"])
            else:
                # Use continuous colormap (gradient)
                if is_zero:
                    color = "grey"
                elif self.vmin < row["trait_value"] < self.vmax:
                    color = mcolors.to_hex(cmap_rev(norm(row["trait_value"])))
                else:
                    color = "black"
            
            self.ax2.plot(
                row["longitude"],
                row["latitude"],
                "o",
                markersize=8,
                markerfacecolor=color,
                markeredgewidth=2,
                markeredgecolor="black",
                alpha=0.2 if is_zero else 1.0,
            )

        # Add trait colorbar (skip if legend is shown)
        if not self.legend_config['show_legend']:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_rev)
            cbar = self.fig.colorbar(
                sm,
                ax=self.ax_tree,
                orientation="vertical",
                pad=0.02,
                location="left",
                shrink=0.9,
                aspect=30,
            )
            cbar.set_label(self.trait_name, fontsize=20)
            cbar.ax.tick_params(labelsize=18)
            
            if self.verbose:
                print(f"  ✓ Colorbar added")

        # ============================
        # CONNECTION LINES
        # ============================
        
        # Build colormap for continuous gradient (if not already built for non-categorical mode)
        if not (self.color_map and self.legend_config.get('show_legend', False)):
            # Only build if we haven't already (to avoid duplicate code)
            base = plt.get_cmap("viridis")
            newcolors = base(np.linspace(0, 0.8, 256))
            viridis_no_yellow = mcolors.ListedColormap(newcolors, name="viridis_no_yellow")
            cmap_rev = viridis_no_yellow.reversed()
            norm = mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)

        # Group GPS data by specimen without assuming "color" column exists
        gps_grouped = (
            self.gps.groupby("specimen_id")[["longitude", "latitude", "trait_value"]]
            .apply(lambda x: list(zip(x["longitude"], x["latitude"], x["trait_value"])))
            .to_dict()
        )

        for index, row in df.iterrows():
            specimen = row["ID"]
            if specimen in gps_grouped:
                for longitude, latitude, trait_val in gps_grouped[specimen]:
                    # Use categorical colors if legend is enabled, otherwise use gradient
                    if self.color_map and self.legend_config.get('show_legend', False):
                        color = self._get_color_from_trait(trait_val)
                    else:
                        # Use continuous colormap (gradient)
                        is_zero = trait_val == 0.0
                        if is_zero:
                            color = "grey"
                        elif self.vmin < trait_val < self.vmax:
                            color = mcolors.to_hex(cmap_rev(norm(trait_val)))
                        else:
                            color = "black"
                    
                    lw = 2.0 if trait_val > 0 else 0.8
                    # Use higher alpha for gradient lines, keep grey semi-transparent
                    line_alpha = 0.3 if color == "grey" else 0.5

                    con = ConnectionPatch(
                        xyA=row["Coordinates"],
                        coordsA="data",
                        xyB=(longitude, latitude),
                        coordsB="data",
                        axesA=self.ax_tree,
                        axesB=self.ax2,
                        color=color,
                        linewidth=lw,
                        linestyle="--",
                        alpha=line_alpha,
                        zorder=2,
                    )
                    self.fig.add_artist(con)

        if self.verbose:
            print(f"  ✓ Connection lines drawn")

        # ============================
        # CATEGORICAL LEGEND (Optional)
        # ============================
        if self.legend_config['show_legend']:
            # Use provided color_to_label mapping
            if not self.color_to_label:
                raise ValueError(
                    "legend_config['show_legend']=True but color_to_label is not provided. "
                    "Please provide color_to_label dict mapping colors to labels."
                )
            
            # Create legend handles from provided color_to_label mapping
            legend_handles = [
                mlines.Line2D(
                    [], [],
                    color=color,
                    marker="o",
                    markersize=self.legend_config['marker_size'],
                    markeredgecolor=self.legend_config['marker_edge_color'],
                    linestyle="None",
                    label=label
                )
                for color, label in self.color_to_label.items()
            ]
            
            # Add legend to map axes
            legend = self.ax2.legend(
                handles=legend_handles,
                title=self.legend_config['title'],
                loc=self.legend_config['loc'],
                bbox_to_anchor=self.legend_config['bbox_to_anchor'],
                fontsize=self.legend_config['fontsize'],
                title_fontsize=self.legend_config['title_fontsize'],
                frameon=self.legend_config['frameon'],
                markerscale=self.legend_config['markerscale'],
                framealpha=self.legend_config['framealpha'],
                edgecolor=self.legend_config['edgecolor'],
                fancybox=self.legend_config['fancybox'],
                shadow=self.legend_config['shadow'],
            )
            
            # Explicitly set legend position for Cartopy axes
            # (bbox_to_anchor alone doesn't always work with Cartopy)
            if legend:
                # Position in figure coordinates (0-1 range)
                # Center below the map panel, positioned higher
                legend.set_bbox_to_anchor(
                    (0.70, 0.12),  # Below map, higher up from bottom
                    transform=self.fig.transFigure
                )
                # Prevent legend from being clipped
                legend.set_clip_on(False)
            
            if self.verbose:
                print(f"  ✓ Categorical legend added at bottom left")

        # Adjust spacing based on whether we have raster or map
        if self.raster_data is not None:
            # Raster mode: space for colorbar on right
            self.fig.subplots_adjust(wspace=0.2, bottom=0.08, left=0.15)
        else:
            # Map mode: legend will overlay, no extra bottom space needed
            self.fig.subplots_adjust(wspace=0.05, bottom=0.08, left=0.1)

        # NOTE: Removed plt.tight_layout() as it overrides legend positioning
        # Use subplots_adjust() instead for better control

        return self.fig

    def save(self, output_dir=None, prefix="tree"):
        """Save the figure to SVG and PNG."""
        if self.fig is None:
            raise ValueError("Plot has not been created yet. Call plot() first.")

        output_dir = (
            Path(output_dir)
            if output_dir
            else Path(self.gps_file).parent.parent / "output"
        )
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_raster" if self.raster_data is not None else "_map"
        svg_file = output_dir / f"{prefix}_to{suffix}_{timestamp}.svg"
        png_file = output_dir / f"{prefix}_to{suffix}_{timestamp}.png"

        self.fig.savefig(
            str(svg_file),
            format="svg",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=300,
        )
        if self.verbose:
            print(f"\n✓ Saved SVG: {svg_file}")

        self.fig.savefig(
            str(png_file),
            format="png",
            # transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=300,
        )
        if self.verbose:
            print(f"✓ Saved PNG: {png_file}")

        return svg_file, png_file

    def show(self):
        """Display the figure."""
        if self.fig is None:
            raise ValueError("Plot has not been created yet. Call plot() first.")
        plt.show()

    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="PhyloCartoPlot: Create phylogeographic visualizations with raster layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With raster (high contrast)
  python tree_to_map_raster.py --nwk tree.nwk --gps gps.csv --offset offsets.csv \\
      --raster enviro.tif --raster-band 1
  
  # With categorical legend (Cumacea)
  python tree_to_map_raster.py --nwk tree.nwk --gps gps.csv --offset offsets.csv \\
      --color-map-file colors.json --show-legend \\
      --legend-title "Marine Basins"
  
  # Custom extent and trait range
  python tree_to_map_raster.py --nwk tree.nwk --gps gps.csv --offset offsets.csv \\
      --extent 43 51 -27 -11 --vmin 0.01 --vmax 0.06 --trait-name "Trait (%)"
        """,
    )

    parser.add_argument(
        "--nwk",
        type=str,
        required=True,
        help="Path to phylogenetic tree (Newick format)",
    )
    parser.add_argument(
        "--gps",
        type=str,
        required=True,
        help="Path to GPS coordinates with trait values",
    )
    parser.add_argument(
        "--offset", type=str, required=True, help="Path to node position adjustments"
    )
    parser.add_argument(
        "--raster", type=str, default=None, help="Path to raster file (GeoTIFF)"
    )
    parser.add_argument(
        "--raster-band",
        type=int,
        default=1,
        help="Band number in raster file (default: 1)",
    )
    parser.add_argument(
        "--raster-metadata",
        type=str,
        default=None,
        help="Path to raster metadata JSON file",
    )
    parser.add_argument(
        "--raster-contrast",
        type=str,
        default="percentile",
        choices=["percentile", "histogram_eq", "sigmoid"],
        help="Contrast enhancement method (default: percentile)",
    )
    parser.add_argument(
        "--raster-cmap",
        type=str,
        default="terrain",
        help="Colormap for raster (default: terrain)",
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=[43, 51, -27, -11],
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
        help="Map extent (default: Madagascar)",
    )
    parser.add_argument(
        "--vmin", type=float, default=None, help="Trait colormap minimum"
    )
    parser.add_argument(
        "--vmax", type=float, default=None, help="Trait colormap maximum"
    )
    parser.add_argument(
        "--trait-name", type=str, default="Trait Value", help="Name of the trait"
    )
    parser.add_argument(
        "--color-map-file",
        type=str,
        default=None,
        help="Path to JSON file with color_map and color_to_label dictionaries"
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Show categorical legend instead of continuous colorbar"
    )
    parser.add_argument(
        "--legend-title",
        type=str,
        default=None,
        help="Title for the categorical legend"
    )
    parser.add_argument(
        "--legend-loc",
        type=str,
        default="lower right",
        help="Legend position (default: lower right)"
    )
    parser.add_argument(
        "--legend-fontsize",
        type=int,
        default=10,
        help="Legend font size (default: 10)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Load color mappings from JSON file if provided
    color_map = None
    color_to_label = None
    
    if args.color_map_file:
        try:
            with open(args.color_map_file, 'r') as f:
                config = json.load(f)
            
            # Extract color_map and color_to_label from JSON
            if 'color_map' in config:
                color_map = {int(k): v for k, v in config['color_map'].items()}
            if 'color_to_label' in config:
                color_to_label = config['color_to_label']
            
            print(f"✓ Loaded color mappings from {args.color_map_file}")
        except Exception as e:
            print(f"✗ Error loading color map file: {e}")
            raise
    
    # Set up legend config
    legend_config = None
    if args.show_legend:
        if not color_to_label:
            raise ValueError(
                "--show-legend requires --color-map-file with color_to_label"
            )
        legend_config = {
            'show_legend': True,
            'title': args.legend_title,
            'loc': args.legend_loc,
            'fontsize': args.legend_fontsize,
        }

    # Create plotter
    plotter = PhyloCartoPlotter(
        nwk_file=args.nwk,
        gps_file=args.gps,
        offset_file=args.offset,
        raster_file=args.raster,
        raster_band=args.raster_band,
        raster_metadata_file=args.raster_metadata,
        raster_contrast=args.raster_contrast,
        raster_cmap=args.raster_cmap,
        extent=args.extent,
        vmin=args.vmin,
        vmax=args.vmax,
        trait_name=args.trait_name,
        color_map=color_map,
        color_to_label=color_to_label,
        legend_config=legend_config,
        verbose=True,
    )

    # Plot and save
    plotter.plot()
    plotter.save(output_dir=args.output)

    print(f"\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()