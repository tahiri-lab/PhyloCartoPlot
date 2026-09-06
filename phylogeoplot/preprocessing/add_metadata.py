import pandas as pd
import argparse

#input_file = '../data/gbif_coffea_ex3_formatted.csv'
#metadata_file = '../data/no_caffeine_nodes_w_specimen.csv'

def add_metadata(input_file, metadata_file):
    
    gbif_df = pd.read_csv(input_file)
    node_names_df = pd.read_csv(metadata_file)

    # Merge the two DataFrames based on 'specimen_id' in gbif_df and 'Species_name' in node_names_df
    merged_df = pd.merge(gbif_df, node_names_df[['Species_name', 'trait_value']], 
                         left_on='specimen_id', right_on='Species_name', how='left')

    # Drop the 'Species_name' column as it's no longer needed
    merged_df = merged_df.drop(columns=['Species_name'])
    
    # Retain only the required columns
    merged_df = merged_df[['specimen_id', 'longitude', 'latitude', 'trait_value']]
    
    # Drop rows where 'trait_value' is NaN
    merged_df = merged_df.dropna(subset=['trait_value'])

    # Create output filename by replacing 'formatted' with 'w_metadata'
    output_file = input_file.replace('formatted', 'w_metadata')

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)

    # Print the output file name
    print(f"Data saved to: {output_file}")
    
    
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process and merge data files.')
    parser.add_argument('input_file', type=str, help='Path to the input GBIF data CSV file.')
    parser.add_argument('metadata_file', type=str, help='Path to the metadata CSV file.')
    args = parser.parse_args()

    # Process the data
    add_metadata(args.input_file, args.metadata_file)

if __name__ == "__main__":
    main()