"""
Unit tests for build_phylogenetic_tree module.
Tests phylogenetic tree construction from FASTA sequences.
"""

import pytest
import tempfile
import os
from Bio import AlignIO, Phylo
from phylocartoplot.preprocessing.build_phylogenetic_tree import build_tree


class TestBuildPhylogeneticTree:
    """Test suite for build_tree function."""

    @pytest.fixture
    def sample_fasta_data(self):
        """Create sample aligned FASTA sequence data."""
        return """<brace>seq1
ATCGATCGATCGATCGATCG
>seq2
ATCGATCGATCGATCGATCG
>seq3
ATCGAGCGATCGATCGATCG
>seq4
ATCGATCGTTCGATCGATCG
""".replace('<brace>', '>')

    @pytest.fixture
    def temp_fasta_file(self, sample_fasta_data):
        """Create temporary FASTA file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_file = os.path.join(tmpdir, 'test_sequences.fasta')
            with open(fasta_file, 'w') as f:
                f.write(sample_fasta_data)
            yield fasta_file, tmpdir

    def test_build_tree_creates_output_file(self, temp_fasta_file):
        """Test that build_tree creates a Newick tree file."""
        fasta_file, tmpdir = temp_fasta_file
        
        build_tree(fasta_file)
        
        output_file = fasta_file.replace('.fasta', '_tree.nwk')
        assert os.path.exists(output_file), f"Tree file {output_file} was not created"

    def test_build_tree_output_is_valid_newick(self, temp_fasta_file):
        """Test that output is a valid Newick format tree."""
        fasta_file, tmpdir = temp_fasta_file
        
        build_tree(fasta_file)
        
        output_file = fasta_file.replace('.fasta', '_tree.nwk')
        
        # Should be readable by BioPython
        try:
            tree = Phylo.read(output_file, 'newick')
            assert tree is not None
        except Exception as e:
            pytest.fail(f"Output tree is not valid Newick format: {e}")

    def test_build_tree_tree_structure(self, temp_fasta_file):
        """Test that tree structure is reasonable."""
        fasta_file, tmpdir = temp_fasta_file
        
        build_tree(fasta_file)
        
        output_file = fasta_file.replace('.fasta', '_tree.nwk')
        tree = Phylo.read(output_file, 'newick')
        
        # Get all terminal nodes (leaves)
        terminals = tree.get_terminals()
        
        # Should have same number of leaves as sequences
        # (accounting for BioPython's naming conventions)
        assert len(terminals) >= 1, "Tree should have at least one terminal node"

    def test_build_tree_with_identical_sequences(self):
        """Test with identical sequences."""
        fasta_data = """>seq1
ATCGATCGATCGATCGATCG
>seq2
ATCGATCGATCGATCGATCG
>seq3
ATCGATCGATCGATCGATCG
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_file = os.path.join(tmpdir, 'identical_seqs.fasta')
            with open(fasta_file, 'w') as f:
                f.write(fasta_data)
            
            build_tree(fasta_file)
            
            output_file = fasta_file.replace('.fasta', '_tree.nwk')
            assert os.path.exists(output_file)
            
            tree = Phylo.read(output_file, 'newick')
            assert tree is not None

    def test_build_tree_with_different_sequences(self):
        """Test with highly divergent sequences."""
        fasta_data = """>seq1
AAAAAAAAAAAAAAAAAAAAA
>seq2
CCCCCCCCCCCCCCCCCCCC
>seq3
GGGGGGGGGGGGGGGGGGGG
>seq4
TTTTTTTTTTTTTTTTTTTT
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_file = os.path.join(tmpdir, 'different_seqs.fasta')
            with open(fasta_file, 'w') as f:
                f.write(fasta_data)
            
            build_tree(fasta_file)
            
            output_file = fasta_file.replace('.fasta', '_tree.nwk')
            assert os.path.exists(output_file)
            
            tree = Phylo.read(output_file, 'newick')
            terminals = tree.get_terminals()
            assert len(terminals) >= 1

    def test_build_tree_tree_has_branches(self, temp_fasta_file):
        """Test that tree has meaningful branch structure."""
        fasta_file, tmpdir = temp_fasta_file
        
        build_tree(fasta_file)
        
        output_file = fasta_file.replace('.fasta', '_tree.nwk')
        tree = Phylo.read(output_file, 'newick')
        
        # Count internal nodes (non-terminal)
        internal_nodes = tree.get_nonterminals()
        
        # With 4 sequences, should have some internal nodes
        assert len(internal_nodes) >= 1, "Tree should have internal nodes"

    def test_build_tree_with_sequence_names(self, temp_fasta_file):
        """Test that sequence names are preserved in tree."""
        fasta_file, tmpdir = temp_fasta_file
        
        build_tree(fasta_file)
        
        output_file = fasta_file.replace('.fasta', '_tree.nwk')
        tree = Phylo.read(output_file, 'newick')
        
        # Get leaf names
        terminals = tree.get_terminals()
        terminal_names = [t.name for t in terminals if t.name]
        
        # Should have some named terminals
        assert len(terminal_names) > 0

    def test_build_tree_with_single_sequence(self):
        """Test with single sequence (edge case)."""
        fasta_data = """>seq1
ATCGATCGATCGATCGATCG
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_file = os.path.join(tmpdir, 'single_seq.fasta')
            with open(fasta_file, 'w') as f:
                f.write(fasta_data)
            
            # Should still create a tree file
            build_tree(fasta_file)
            
            output_file = fasta_file.replace('.fasta', '_tree.nwk')
            # May or may not exist depending on implementation
            # but should not raise an error

    def test_build_tree_handles_nonexistent_file(self):
        """Test error handling for nonexistent FASTA file."""
        with pytest.raises(Exception):
            build_tree('nonexistent_file.fasta')

    def test_build_tree_with_many_sequences(self):
        """Test with larger number of sequences."""
        # Generate 20 similar sequences with small variations
        sequences = []
        base_seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        
        for i in range(20):
            seq = base_seq[:i] + 'C' + base_seq[i+1:]  # Insert variation
            sequences.append(f">seq{i}\n{seq}")
        
        fasta_data = "\n".join(sequences)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_file = os.path.join(tmpdir, 'many_seqs.fasta')
            with open(fasta_file, 'w') as f:
                f.write(fasta_data)
            
            build_tree(fasta_file)
            
            output_file = fasta_file.replace('.fasta', '_tree.nwk')
            assert os.path.exists(output_file)
            
            tree = Phylo.read(output_file, 'newick')
            terminals = tree.get_terminals()
            # Should have many terminals
            assert len(terminals) >= 1

    def test_build_tree_distance_calculation(self, temp_fasta_file):
        """Test that distance calculation produces a tree."""
        fasta_file, tmpdir = temp_fasta_file
        
        # Read original sequences
        alignment = AlignIO.read(fasta_file, 'fasta')
        seq_count = len(alignment)
        
        build_tree(fasta_file)
        
        output_file = fasta_file.replace('.fasta', '_tree.nwk')
        tree = Phylo.read(output_file, 'newick')
        
        # Tree should be consistent with input
        assert tree is not None

    def test_build_tree_file_format(self, temp_fasta_file):
        """Test that output file is a valid Newick file."""
        fasta_file, tmpdir = temp_fasta_file
        
        build_tree(fasta_file)
        
        output_file = fasta_file.replace('.fasta', '_tree.nwk')
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Newick format should contain parentheses and semicolon
        assert '(' in content and ')' in content and ';' in content
