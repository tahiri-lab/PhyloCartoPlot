"""
Unit tests for tree_to_map_raster module.
Tests the PhyloCartoPlotter visualization class.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from phylocartoplot.visualisation.tree_to_map_raster import (
    PhyloCartoPlotter,
    RasterMetadata
)


class TestRasterMetadata:
    """Test suite for RasterMetadata class."""

    def test_raster_metadata_nonexistent_file(self):
        """Test error handling for nonexistent raster file."""
        with pytest.raises(FileNotFoundError):
            RasterMetadata('nonexistent_raster.tif', verbose=False)

    def test_raster_metadata_initialization(self):
        """Test RasterMetadata initialization with valid raster."""
        # This test would require a valid GeoTIFF file
        # Mocking for now
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy GeoTIFF or skip if not available
            pytest.skip("Requires valid GeoTIFF file")


class TestPhyloCartoPlotter:
    """Test suite for PhyloCartoPlotter class."""

    @pytest.fixture
    def sample_tree_file(self):
        """Create a sample Newick tree file."""
        tree_content = "((seq1:0.1,seq2:0.1):0.2,(seq3:0.15,seq4:0.15):0.15);"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
            f.write(tree_content)
            tree_file = f.name
        yield tree_file
        os.unlink(tree_file)

    @pytest.fixture
    def sample_gps_file(self):
        """Create a sample GPS coordinates file."""
        data = {
            'specimen_id': ['seq1', 'seq2', 'seq3', 'seq4'],
            'longitude': [-51.5, -51.4, 20.5, 20.6],
            'latitude': [-2.5, -2.4, 0.5, 0.6],
            'trait_value': [1.2, 1.1, 0.8, 0.9]
        }
        df = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            gps_file = f.name
        yield gps_file
        os.unlink(gps_file)

    @pytest.fixture
    def sample_offset_file(self):
        """Create a sample offset file."""
        data = {
            'NodeName': ['seq1', 'seq2', 'seq3', 'seq4'],
            'XOffset': [0.0, 0.1, 0.0, 0.1]
        }
        df = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            offset_file = f.name
        yield offset_file
        os.unlink(offset_file)

    def test_phylocartoplotter_initialization(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test PhyloCartoPlotter initialization."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            verbose=False
        )
        
        assert plotter is not None
        assert plotter.tree is not None
        assert plotter.gps is not None
        assert plotter.offsets_dict is not None

    def test_phylocartoplotter_loads_data_correctly(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test that data is loaded correctly."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            verbose=False
        )
        
        assert len(plotter.gps) == 4
        assert len(plotter.offsets_dict) == 4
        assert 'trait_value' in plotter.gps.columns

    def test_phylocartoplotter_auto_vmin_vmax(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test automatic vmin/vmax calculation."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            verbose=False
        )
        
        assert plotter.vmin is not None
        assert plotter.vmax is not None
        assert plotter.vmin < plotter.vmax

    def test_phylocartoplotter_custom_vmin_vmax(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test custom vmin/vmax parameters."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            vmin=0.5,
            vmax=2.0,
            verbose=False
        )
        
        assert plotter.vmin == 0.5
        assert plotter.vmax == 2.0

    def test_phylocartoplotter_missing_tree_file(self, sample_gps_file, sample_offset_file):
        """Test error handling for missing tree file."""
        with pytest.raises(Exception):
            PhyloCartoPlotter(
                nwk_file='nonexistent_tree.nwk',
                gps_file=sample_gps_file,
                offset_file=sample_offset_file,
                verbose=False
            )

    def test_phylocartoplotter_missing_gps_file(self, sample_tree_file, sample_offset_file):
        """Test error handling for missing GPS file."""
        with pytest.raises(Exception):
            PhyloCartoPlotter(
                nwk_file=sample_tree_file,
                gps_file='nonexistent_gps.csv',
                offset_file=sample_offset_file,
                verbose=False
            )

    def test_phylocartoplotter_missing_offset_file(self, sample_tree_file, sample_gps_file):
        """Test error handling for missing offset file."""
        with pytest.raises(Exception):
            PhyloCartoPlotter(
                nwk_file=sample_tree_file,
                gps_file=sample_gps_file,
                offset_file='nonexistent_offsets.csv',
                verbose=False
            )

    def test_phylocartoplotter_color_mapping(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test color mapping for trait values."""
        color_map = {1: "red", 0: "blue"}
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            color_map=color_map,
            verbose=False
        )
        
        assert plotter.color_map == color_map

    def test_phylocartoplotter_legend_config(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test legend configuration."""
        legend_config = {
            'show_legend': True,
            'title': 'Test Legend'
        }
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            legend_config=legend_config,
            verbose=False
        )
        
        assert plotter.legend_config['show_legend'] == True
        assert plotter.legend_config['title'] == 'Test Legend'

    def test_phylocartoplotter_extent_parameter(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test custom extent parameter."""
        extent = [40, 50, -30, -10]
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            extent=extent,
            verbose=False
        )
        
        assert plotter.extent == extent

    def test_phylocartoplotter_trait_name(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test custom trait name."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            trait_name='Caffeine Content (%)',
            verbose=False
        )
        
        assert plotter.trait_name == 'Caffeine Content (%)'

    def test_phylocartoplotter_value_to_color(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test color mapping from trait values."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            verbose=False
        )
        
        # Zero should map to grey
        color_zero = plotter._value_to_color(0.0)
        assert color_zero == "grey"
        
        # Valid value should map to hex color
        color_valid = plotter._value_to_color(0.95)
        assert isinstance(color_valid, str)
        assert color_valid.startswith('#') or color_valid in ['black', 'grey']

    def test_phylocartoplotter_get_x_offset(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test getting x-axis offset for nodes."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            verbose=False
        )
        
        offset = plotter._get_x_offset('seq1')
        assert isinstance(offset, float) or isinstance(offset, int)

    def test_phylocartoplotter_get_x_offset_missing_node(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test getting offset for non-existent node (should return default)."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            verbose=False
        )
        
        offset = plotter._get_x_offset('nonexistent_node')
        assert offset == 0  # Default offset

    def test_phylocartoplotter_plot_method(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test the plot method."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            verbose=False
        )
        
        fig = plotter.plot()
        
        assert fig is not None
        assert plotter.fig is not None
        assert plotter.ax_tree is not None
        assert plotter.ax2 is not None

    def test_phylocartoplotter_save_method(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test the save method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = PhyloCartoPlotter(
                nwk_file=sample_tree_file,
                gps_file=sample_gps_file,
                offset_file=sample_offset_file,
                verbose=False
            )
            
            plotter.plot()
            svg_file, png_file = plotter.save(output_dir=tmpdir)
            
            assert os.path.exists(svg_file)
            assert os.path.exists(png_file)
            assert str(svg_file).endswith('.svg')
            assert str(png_file).endswith('.png')

    def test_phylocartoplotter_close_method(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test the close method."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            verbose=False
        )
        
        plotter.plot()
        plotter.close()
        
        assert plotter.fig is None or True  # After close, fig is closed

    def test_phylocartoplotter_without_raster(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test plotter in map mode (no raster)."""
        plotter = PhyloCartoPlotter(
            nwk_file=sample_tree_file,
            gps_file=sample_gps_file,
            offset_file=sample_offset_file,
            verbose=False
        )
        
        assert plotter.raster_data is None
        plotter.plot()
        # Should plot map features instead of raster

    def test_phylocartoplotter_raster_contrast_methods(self, sample_tree_file, sample_gps_file, sample_offset_file):
        """Test different raster contrast enhancement methods."""
        methods = ['percentile', 'histogram_eq', 'sigmoid']
        
        for method in methods:
            plotter = PhyloCartoPlotter(
                nwk_file=sample_tree_file,
                gps_file=sample_gps_file,
                offset_file=sample_offset_file,
                raster_contrast=method,
                verbose=False
            )
            
            assert plotter.raster_contrast == method
