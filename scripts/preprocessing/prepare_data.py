import argparse
import os

from format_gbif_data import format_gbif  
from workflow.scripts.preprocessing.add_metadata import add_metadata
from build_phylogenetic_tree import build_tree

def main():
    
    parser = argparse.ArgumentParser(description='PhyloCartoPlot Main script')
    parser.add_argument('--data', type=str, required=True, help='Path to the file for gbif data')
    parser.add_argument('--nodes', type=str, required=True, help='Path to the file for nodes')
    parser.add_argument('--meta', type=str, required=True, help='Path to the file for add metadata')
    parser.add_argument('--fasta', type=str, required=True, help='Path to the fasta file for tree creation')

    args = parser.parse_args()

    ### FORMAT GBIF DATA WITH COORDINATES AND CAFFEINE CONTENT ###

    print("Calling format gbif with files:", args.data, args.nodes)
    format_gbif(args.data, args.nodes)

    base_name, extension = os.path.splitext(args.data)

    formatted_csv_file = base_name + '_formatted' + extension

    print("Calling add_metadata with files:", formatted_csv_file, args.caf)
    add_metadata(formatted_csv_file, args.caf)

    ### CREATE PHYLOGENETIC TREE ###

    print("Calling build_tree with file:", args.fasta)
    build_tree(args.fasta)
    


if __name__ == "__main__":
    main()

# example use
# python .\phylo_carto_plot.py --data ..\input\gbif_coffea_ex3.csv --nodes ..\input\node_names.csv --meta ..\input\no_caffeine_nodes_w_specimen.csv --fasta ..\input\aligned.fasta