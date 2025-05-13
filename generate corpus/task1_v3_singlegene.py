import pandas as pd

# Define the input and output file paths
input_csv_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\combined.csv'
output_parquet_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\1031\task1_v3_singlegene.parquet'

# Load the CSV file and remove duplicates
df = pd.read_csv(input_csv_file)
df = df.drop_duplicates(subset=['Gene ID', 'Chromosome', 'Start', 'End', 'Strand'])
print(df.head())

# Initialize lists for prompts, inputs, and responses
prompts = []
inputs = []
responses = []
prompt_t = "You are a bioinformatics expert. Answer the following question accurately and professionally."

# Generate questions, prompts, and responses for each row in the DataFrame
for index, row in df.iterrows():
    gene_symbol = row['Gene Symbol']
    gene_id = str(row["Gene ID"])
    chromosome = str(row['Chromosome'])
    start = str(row['Start'])
    end = str(row['End'])
    
    # Generate different question templates for each gene attribute
    inputs.extend([
        # # Gene ID related questions
        # f"What is the Gene ID for {gene_symbol}?",
        # f"Can you specify the Gene ID of {gene_symbol}?",
        # f"Identify the Gene ID for {gene_symbol}.",
        # f"Provide the Gene ID associated with {gene_symbol}.",
        # f"Which Gene ID corresponds to the gene {gene_symbol}?",

        # Chromosome location related questions
        f"Where is the gene {gene_symbol} located in the human chromosome?",
        f"On which chromosome is {gene_symbol} located?",
        f"{gene_symbol} is mapped to which chromosome?",
        f"In which chromosome can {gene_symbol} be found?",
        f"Identify the chromosome for {gene_symbol}.",
        
        # Start position related questions
        f"What is the start position of {gene_symbol} on chromosome {chromosome}?",
        f"Where does {gene_symbol} start on chromosome {chromosome}?",
        f"At which position does {gene_symbol} start on chromosome {chromosome}?",
        f"Provide the starting base pair position for {gene_symbol} on chromosome {chromosome}.",
        f"Identify the start position of {gene_symbol} on chromosome {chromosome}.",
        
        # End position related questions
        f"What is the end position of {gene_symbol} on chromosome {chromosome}?",
        f"Where does {gene_symbol} end on chromosome {chromosome}?",
        f"At which position does {gene_symbol} end on chromosome {chromosome}?",
        f"Provide the ending base pair position for {gene_symbol} on chromosome {chromosome}.",
        f"Identify the end position of {gene_symbol} on chromosome {chromosome}."
    ])
    
    # Generate corresponding responses for each question type
    responses.extend([
        # # Gene ID responses
        # gene_id, gene_id, gene_id, gene_id, gene_id,
        
        # Chromosome responses
        chromosome, chromosome, chromosome, chromosome, chromosome, 
        
        # Start position responses
        start, start, start, start, start, 
        
        # End position responses
        end, end, end, end, end,
    ])
    
    # Add the prompt for each question
    prompts.extend([prompt_t] * 15)

# Create a DataFrame with the prompts, inputs, and responses
df_fine_tuning = pd.DataFrame({
    'Prompt': prompts,
    'Input': inputs,
    'Response': responses
})

# Save the DataFrame to a parquet file
df_fine_tuning.to_parquet(output_parquet_file, index=False)

print(f"Parquet file saved to: {output_parquet_file}")
