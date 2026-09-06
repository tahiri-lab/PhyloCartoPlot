"""
Unit tests for add_metadata module.
Tests merging of trait/metadata values with geographic data.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from phylogeoplot.preprocessing.add_metadata import add_metadata


class TestAddMetadata:
    """Test suite for add_metadata function."""

    @pytest.fixture
    def sample_formatted_coords(self):
        """Create sample formatted geographic data."""
        data = {
            'specimen_id': ['coffea_arabica', 'coffea_robusta', 'coffea_canephora'],
            'longitude': [-51.5, 20.5, 25.0],
            'latitude': [-2.5, 0.5, 1.0]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata with trait values."""
        data = {
            'Species_name': ['coffea_arabica', 'coffea_robusta', 'coffea_canephora'],
            'trait_value': [1.2, 0.8, 0.95]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_csv_files(self, sample_formatted_coords, sample_metadata):
        """Create temporary CSV files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coords_file = os.path.join(tmpdir, 'test_coords_formatted.csv')
            meta_file = os.path.join(tmpdir, 'test_metadata.csv')
            
            sample_formatted_coords.to_csv(coords_file, index=False)
            sample_metadata.to_csv(meta_file, index=False)
            
            yield coords_file, meta_file, tmpdir

    def test_add_metadata_output_file_created(self, temp_csv_files):
        """Test that add_metadata creates an output file."""
        coords_file, meta_file, tmpdir = temp_csv_files
        
        add_metadata(coords_file, meta_file)
        
        output_file = coords_file.replace('formatted', 'w_metadata')
        assert os.path.exists(output_file), f"Output file {output_file} was not created"

    def test_add_metadata_output_has_required_columns(self, temp_csv_files):
        """Test that output has specimen_id, coordinates, and trait_value."""
        coords_file, meta_file, tmpdir = temp_csv_files
        
        add_metadata(coords_file, meta_file)
        
        output_file = coords_file.replace('formatted', 'w_metadata')
        output_df = pd.read_csv(output_file)
        
        required_columns = ['specimen_id', 'longitude', 'latitude', 'trait_value']
        assert all(col in output_df.columns for col in required_columns)

    def test_add_metadata_merge_correctness(self, temp_csv_files):
        """Test that metadata is correctly merged by specimen_id."""
        coords_file, meta_file, tmpdir = temp_csv_files
        
        add_metadata(coords_file, meta_file)
        
        output_file = coords_file.replace('formatted', 'w_metadata')
        output_df = pd.read_csv(output_file)
        metadata_df = pd.read_csv(meta_file)
        
        # Verify each record has correct trait value
        for _, row in output_df.iterrows():
            specimen = row['specimen_id']
            expected_trait = metadata_df[
                metadata_df['Species_name'] == specimen
            ]['trait_value'].values[0]
            assert row['trait_value'] == expected_trait

    def test_add_metadata_removes_unmatched_records(self):
        """Test that records without metadata are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coords_data = pd.DataFrame({
                'specimen_id': ['species_a', 'species_b', 'species_c'],
                'longitude': [10.0, 20.0, 30.0],
                'latitude': [5.0, 10.0, 15.0]
            })
            
            # Metadata only for two species
            meta_data = pd.DataFrame({
                'Species_name': ['species_a', 'species_b'],
                'trait_value': [1.0, 2.0]
            })
            
            coords_file = os.path.join(tmpdir, 'coords_formatted.csv')
            meta_file = os.path.join(tmpdir, 'metadata.csv')
            
            coords_data.to_csv(coords_file, index=False)
            meta_data.to_csv(meta_file, index=False)
            
            add_metadata(coords_file, meta_file)
            
            output_file = coords_file.replace('formatted', 'w_metadata')
            output_df = pd.read_csv(output_file)
            
            # Only matched records should remain
            assert len(output_df) == 2
            assert 'species_c' not in output_df['specimen_id'].values

    def test_add_metadata_handles_missing_trait_values(self):
        """Test handling of NaN trait values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coords_data = pd.DataFrame({
                'specimen_id': ['species_a', 'species_b'],
                'longitude': [10.0, 20.0],
                'latitude': [5.0, 10.0]
            })
            
            # One record with NaN trait_value
            meta_data = pd.DataFrame({
                'Species_name': ['species_a', 'species_b'],
                'trait_value': [1.0, float('nan')]
            })
            
            coords_file = os.path.join(tmpdir, 'coords_formatted.csv')
            meta_file = os.path.join(tmpdir, 'metadata.csv')
            
            coords_data.to_csv(coords_file, index=False)
            meta_data.to_csv(meta_file, index=False)
            
            add_metadata(coords_file, meta_file)
            
            output_file = coords_file.replace('formatted', 'w_metadata')
            output_df = pd.read_csv(output_file)
            
            # Record with NaN should be removed
            assert len(output_df) == 1
            assert output_df.iloc[0]['specimen_id'] == 'species_a'

    def test_add_metadata_preserves_coordinates(self, temp_csv_files):
        """Test that coordinates are preserved correctly."""
        coords_file, meta_file, tmpdir = temp_csv_files
        original_coords = pd.read_csv(coords_file)
        
        add_metadata(coords_file, meta_file)
        
        output_file = coords_file.replace('formatted', 'w_metadata')
        output_df = pd.read_csv(output_file)
        
        # Check coordinates are preserved
        for _, row in output_df.iterrows():
            original_row = original_coords[
                original_coords['specimen_id'] == row['specimen_id']
            ].iloc[0]
            assert row['longitude'] == original_row['longitude']
            assert row['latitude'] == original_row['latitude']

    def test_add_metadata_with_zero_trait_values(self):
        """Test that zero trait values are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coords_data = pd.DataFrame({
                'specimen_id': ['species_a', 'species_b'],
                'longitude': [10.0, 20.0],
                'latitude': [5.0, 10.0]
            })
            
            meta_data = pd.DataFrame({
                'Species_name': ['species_a', 'species_b'],
                'trait_value': [0.0, 1.5]
            })
            
            coords_file = os.path.join(tmpdir, 'coords_formatted.csv')
            meta_file = os.path.join(tmpdir, 'metadata.csv')
            
            coords_data.to_csv(coords_file, index=False)
            meta_data.to_csv(meta_file, index=False)
            
            add_metadata(coords_file, meta_file)
            
            output_file = coords_file.replace('formatted', 'w_metadata')
            output_df = pd.read_csv(output_file)
            
            # Both records including zero trait value should be included
            assert len(output_df) == 2
            assert any(output_df['trait_value'] == 0.0)

    def test_add_metadata_handles_missing_files(self):
        """Test error handling for missing input files."""
        with pytest.raises(Exception):
            add_metadata('nonexistent_coords.csv', 'nonexistent_meta.csv')

    def test_add_metadata_numeric_trait_values(self):
        """Test with various numeric trait value formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coords_data = pd.DataFrame({
                'specimen_id': ['sp1', 'sp2', 'sp3', 'sp4'],
                'longitude': [10.0, 20.0, 30.0, 40.0],
                'latitude': [5.0, 10.0, 15.0, 20.0]
            })
            
            meta_data = pd.DataFrame({
                'Species_name': ['sp1', 'sp2', 'sp3', 'sp4'],
                'trait_value': [0.001, 50.5, 99.999, 1.0]  # Different scales
            })
            
            coords_file = os.path.join(tmpdir, 'coords_formatted.csv')
            meta_file = os.path.join(tmpdir, 'metadata.csv')
            
            coords_data.to_csv(coords_file, index=False)
            meta_data.to_csv(meta_file, index=False)
            
            add_metadata(coords_file, meta_file)
            
            output_file = coords_file.replace('formatted', 'w_metadata')
            output_df = pd.read_csv(output_file)
            
            # All records should be present with correct trait values
            assert len(output_df) == 4
            assert output_df['trait_value'].min() == 0.001
            assert output_df['trait_value'].max() == 99.999
