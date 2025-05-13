import pandas as pd

# Define the input and output file paths
input_csv_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\combined.csv'
output_parquet_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\1031\task1_v3_multigene.parquet'

# Load the CSV file and remove duplicates
df = pd.read_csv(input_csv_file)
df = df.drop_duplicates(subset=['Gene ID', 'Chromosome', 'Start', 'End', 'Strand'])

# Ensure 'Gene Symbol' column exists
if 'Gene Symbol' not in df.columns:
    raise ValueError("The 'Gene Symbol' column is missing from the data.")

# Initialize lists for prompts, inputs, and responses
prompts = []
inputs = []
responses = []
prompt_t = "You are a bioinformatics expert. Based on the following instruction, provide an accurate and professional response."

# Define the number of repetitions
num_repeats = 30
group_size = 5

for _ in range(num_repeats):
    # Shuffle the DataFrame to ensure randomness for each iteration
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Process DataFrame in groups of 5
    for i in range(0, len(df) - group_size + 1, group_size):
        selected_genes = df.iloc[i:i + group_size].reset_index(drop=True)
        
        # Collect gene information in the required format
        response_for_gene_id = {row['Gene Symbol']: str(row['Gene ID']) for _, row in selected_genes.iterrows()}
        response_for_chromosome = {row['Gene Symbol']: str(row['Chromosome']) for _, row in selected_genes.iterrows()}
        response_for_start = {row['Gene Symbol']: str(row['Start']) for _, row in selected_genes.iterrows()}
        response_for_end = {row['Gene Symbol']: str(row['End']) for _, row in selected_genes.iterrows()}

        # Convert dictionary format to the desired "{gene_symbol}: {value}" structure
        response_gene_id = "; ".join([f"{gene}: {gene_id}" for gene, gene_id in response_for_gene_id.items()])
        response_chromosome = "; ".join([f"{gene}: {chromosome}" for gene, chromosome in response_for_chromosome.items()])
        response_start = "; ".join([f"{gene}: {start}" for gene, start in response_for_start.items()])
        response_end = "; ".join([f"{gene}: {end}" for gene, end in response_for_end.items()])

        # Generate input question templates for this group
        gene_symbols = ", ".join(selected_genes['Gene Symbol'])
        inputs.extend([
            # f"List the Gene IDs for the following genes: {gene_symbols}.",
            # f"Can you specify the Gene IDs for these genes: {gene_symbols}?",
            # f"Provide the Gene IDs for {gene_symbols}.",
            # f"What are the Gene IDs associated with {gene_symbols}?",
            
            f"On which chromosomes are the following genes located: {gene_symbols}?",
            f"List the chromosome locations for these genes: {gene_symbols}.",
            f"Identify the chromosomes for {gene_symbols}.",
            f"Where are {gene_symbols} located in terms of chromosomes?",
            
            f"What are the start positions of {gene_symbols} on their respective chromosomes?",
            f"Provide the starting positions for {gene_symbols}.",
            f"Identify the start locations of {gene_symbols}.",
            f"List the base pair start positions of the genes {gene_symbols}.",
            
            f"What are the end positions of {gene_symbols} on their respective chromosomes?",
            f"Provide the ending positions for {gene_symbols}.",
            f"Identify the end locations of {gene_symbols}.",
            f"List the base pair end positions of the genes {gene_symbols}."
        ])

        # Add the appropriate response for each question type
        responses.extend([
            # response_gene_id,     # Gene IDs for each gene
            # response_gene_id,
            # response_gene_id,
            # response_gene_id,
            
            response_chromosome,  # Chromosome locations for each gene
            response_chromosome,
            response_chromosome,
            response_chromosome,
            
            response_start,       # Start positions for each gene
            response_start,
            response_start,
            response_start,
            
            response_end,         # End positions for each gene
            response_end,
            response_end,
            response_end
        ])
        prompts.extend([prompt_t] * 12)

# Create a DataFrame with the prompts, inputs, and responses
df_fine_tuning = pd.DataFrame({
    'Prompt': prompts,
    'Input': inputs,
    'Response': responses
})

# Save the DataFrame to a parquet file
df_fine_tuning.to_parquet(output_parquet_file, index=False)

print(f"Parquet file saved to: {output_parquet_file}")
