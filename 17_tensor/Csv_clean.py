import pandas as pd
import argparse


def remove_duplicates_with_y_chromosome(df):
    # Find duplicate genes with more than one occurrence
    duplicate_genes = df[df.duplicated(subset='Gene', keep=False)]

    # Remove rows with 'Y' chromosome for duplicate genes
    df = df[~((df['Gene'].isin(duplicate_genes['Gene'])) & (df['Chromosome'] == 'Y'))]

    return df

def main(results_path):
    # Fixed file paths for other files
    pybiomart_path = 'Filter/CSV/pybiomart_q.csv'
    sfari_path = 'Filter/CSV/SFARI-Gene_genes_01-16-2024release_01-19-2024export.csv'
    output_path = results_path.replace('.csv', '_clean.csv')

    # Read data from specified paths
    pybiomart = pd.read_csv(pybiomart_path,)
    results = pd.read_csv(results_path)
    sfari = pd.read_csv(sfari_path)

    # Merge the dataframes
    results = pd.merge(results, pybiomart, left_on='Ensemble_ID', right_on='Gene stable ID')
    results = results[['Gene', 'Ensemble_ID', 'Probability_Class_0', 'Probability_Class_1', 'Chromosome/scaffold name']]

    # Add the SFARI gene scores
    results = pd.merge(results, sfari, left_on='Ensemble_ID', right_on='ensembl-id', how='left')
    results = results[['Gene', 'Ensemble_ID', 'Probability_Class_0', 'Probability_Class_1', 'Chromosome/scaffold name', 'gene-score']]

    # Rename the columns
    results.columns = ['Gene', 'Ensembl_ID', 'Probability_Class_0', 'Probability_Class_1', 'Chromosome', 'SFARI_Gene_Score']

    # Remove duplicates with 'Y' chromosome
    results = remove_duplicates_with_y_chromosome(results)

    # Remove other duplicates
    results=results.drop_duplicates(subset='Gene', keep='first').reset_index(drop=True)
    results=results.sort_values(by="Probability_Class_1",ascending=False).reset_index(drop=True)


    # Save the results
    results.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process results dataframe.')
    parser.add_argument('results', type=str, help='Path to results CSV file')

    args = parser.parse_args()

    main(args.results)
