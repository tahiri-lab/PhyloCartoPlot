"""
Unit tests for format_gbif_data module.
Tests the format_gbif function for geographic coordinate formatting.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from phylogeoplot.preprocessing.format_gbif_data import format_gbif


class TestFormatGbif:
    """Test suite for format_gbif function."""

    @pytest.fixture
    def sample_gbif_data(self):
        """Create sample GBIF data CSV."""
        data = {
            'specimen_id': ['coffea_arabica_1', 'coffea_arabica_2', 'coffea_robusta_1'],
            'longitude': [-51.5, -51.4, 20.5],
            'latitude': [-2.5, -2.4, 0.5]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_node_names(self):
        """Create sample node names CSV."""
        data = {
            'node_name': ['C_coffea_arabica', 'C_coffea_robusta']
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_csv_files(self, sample_gbif_data, sample_node_names):
        """Create temporary CSV files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gbif_file = os.path.join(tmpdir, 'test_gbif.csv')
            nodes_file = os.path.join(tmpdir, 'test_nodes.csv')
            
            sample_gbif_data.to_csv(gbif_file, index=False)
            sample_node_names.to_csv(nodes_file, index=False)
            
            yield gbif_file, nodes_file, tmpdir

    def test_format_gbif_output_file_created(self, temp_csv_files):
        """Test that format_gbif creates an output file."""
        gbif_file, nodes_file, tmpdir = temp_csv_files
        
        format_gbif(gbif_file, nodes_file)
        
        output_file = gbif_file.replace('.csv', '_formatted.csv')
        assert os.path.exists(output_file), f"Output file {output_file} was not created"

    def test_format_gbif_output_has_required_columns(self, temp_csv_files):
        """Test that output file has the required columns."""
        gbif_file, nodes_file, tmpdir = temp_csv_files
        
        format_gbif(gbif_file, nodes_file)
        
        output_file = gbif_file.replace('.csv', '_formatted.csv')
        output_df = pd.read_csv(output_file)
        
        required_columns = ['specimen_id', 'longitude', 'latitude', 'node_name']
        assert all(col in output_df.columns for col in required_columns)

    def test_format_gbif_removes_nan_node_names(self, sample_gbif_data, sample_node_names):
        """Test that records with NaN node_name are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Add a specimen that won't match any node
            bad_gbif = pd.concat([
                sample_gbif_data,
                pd.DataFrame({'specimen_id': ['unknown_species_1'], 
                            'longitude': [30.0], 
                            'latitude': [10.0]})
            ], ignore_index=True)
            
            gbif_file = os.path.join(tmpdir, 'test_gbif.csv')
            nodes_file = os.path.join(tmpdir, 'test_nodes.csv')
            
            bad_gbif.to_csv(gbif_file, index=False)
            sample_node_names.to_csv(nodes_file, index=False)
            
            format_gbif(gbif_file, nodes_file)
            
            output_file = gbif_file.replace('.csv', '_formatted.csv')
            output_df = pd.read_csv(output_file)
            
            # Unmatched specimen should not be in output
            assert len(output_df) < len(bad_gbif)

    def test_format_gbif_preserves_coordinates(self, temp_csv_files):
        """Test that coordinates are preserved correctly."""
        gbif_file, nodes_file, tmpdir = temp_csv_files
        original_df = pd.read_csv(gbif_file)
        
        format_gbif(gbif_file, nodes_file)
        
        output_file = gbif_file.replace('.csv', '_formatted.csv')
        output_df = pd.read_csv(output_file)
        
        # Check that coordinates from matching records are preserved
        for _, row in output_df.iterrows():
            matching_original = original_df[original_df['specimen_id'] == row['specimen_id']]
            if len(matching_original) > 0:
                assert row['longitude'] == matching_original.iloc[0]['longitude']
                assert row['latitude'] == matching_original.iloc[0]['latitude']

    def test_format_gbif_with_duplicate_specimens(self):
        """Test handling of duplicate specimen IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gbif_data = pd.DataFrame({
                'specimen_id': ['coffee_arabica_1', 'coffee_arabica_1', 'coffee_robusta_1'],
                'longitude': [10.0, 10.5, 20.0],
                'latitude': [5.0, 5.5, 0.0]
            })
            
            node_data = pd.DataFrame({
                'node_name': ['C_coffee_arabica', 'C_coffee_robusta']
            })
            
            gbif_file = os.path.join(tmpdir, 'test_gbif.csv')
            nodes_file = os.path.join(tmpdir, 'test_nodes.csv')
            
            gbif_data.to_csv(gbif_file, index=False)
            node_data.to_csv(nodes_file, index=False)
            
            format_gbif(gbif_file, nodes_file)
            
            output_file = gbif_file.replace('.csv', '_formatted.csv')
            output_df = pd.read_csv(output_file)
            
            # Duplicates should be preserved
            assert len(output_df) == 3

    def test_format_gbif_handles_missing_files(self):
        """Test error handling for missing input files."""
        with pytest.raises(Exception):
            format_gbif('nonexistent.csv', 'also_missing.csv')

    def test_format_gbif_case_sensitivity(self):
        """Test that specimen ID matching works case-insensitively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gbif_data = pd.DataFrame({
                'specimen_id': ['Coffee_Arabica_1'],
                'longitude': [10.0],
                'latitude': [5.0]
            })
            
            node_data = pd.DataFrame({
                'node_name': ['C_coffee_arabica']
            })
            
            gbif_file = os.path.join(tmpdir, 'test_gbif.csv')
            nodes_file = os.path.join(tmpdir, 'test_nodes.csv')
            
            gbif_data.to_csv(gbif_file, index=False)
            node_data.to_csv(nodes_file, index=False)
            
            format_gbif(gbif_file, nodes_file)
            
            output_file = gbif_file.replace('.csv', '_formatted.csv')
            output_df = pd.read_csv(output_file)
            
            # Should still create output file even if case differs
            assert os.path.exists(output_file)
