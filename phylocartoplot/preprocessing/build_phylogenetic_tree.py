import sys
from Bio import SeqIO, AlignIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import matplotlib.pyplot as plt
import os

def custom_label(clade):
    if clade.is_terminal():
        return clade.name 
    else:
        return None  # Return None for non-terminal nodes

def build_tree(fasta_file):
    # Step 1: Parse sequences from the FASTA file
    try:
        alignment = AlignIO.read(fasta_file, "fasta")
    except ValueError as error:
        if "same length" not in str(error):
            raise

        records = list(SeqIO.parse(fasta_file, "fasta"))
        if not records:
            raise
        max_length = max(len(record.seq) for record in records)
        for record in records:
            record.seq = record.seq + ("-" * (max_length - len(record.seq)))
        alignment = AlignIO.MultipleSeqAlignment(records)

    # Step 2: Calculate distance matrix
    calculator = DistanceCalculator('identity')
    distance_matrix = calculator.get_distance(alignment)

    # Step 3: Build phylogenetic tree
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(distance_matrix)
    
    base_name, extension = os.path.splitext(fasta_file)
    output_tree_file = base_name + '_tree.nwk'

    # Step 4: Save tree to Newick file
    Phylo.write(tree, output_tree_file, "newick")

    # Step 5: Visualize the tree
    fig, ax = plt.subplots(figsize=(20, 10))

    # Draw the tree
    Phylo.draw(tree, axes=ax, label_func=custom_label)

    # Show the plot
    #plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_phylogenetic_tree.py <input_fasta_file>")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
 
    
    build_tree(fasta_file)
