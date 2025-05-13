import pandas as pd

# Define the input and output file paths
input_csv_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\combined.csv'
output_parquet_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\1031\task1_v3_mask.parquet'

# Load the CSV file and remove duplicates
df = pd.read_csv(input_csv_file).drop_duplicates(subset=['Gene ID', 'Chromosome', 'Start', 'End', 'Strand'])
print(df.head())

# Initialize lists for prompts, inputs, and responses
prompts = []
inputs = []
responses = []

# Define templates for input prompts and corresponding responses
templates = [
    # ("Gene ID for {symbol} in human is <mask>", "{gene_id}"),
    # ("What is the Gene ID of {symbol} in humans? It is <mask>", "{gene_id}"),
    # ("Provide the Gene ID for {symbol}: <mask>", "{gene_id}"),
    ("The chromosome location of {symbol} is <mask>.", "{chromosome}"),
    ("{symbol} in humans can be found on chromosome <mask>", "{chromosome}"),
    ("{symbol} in human is located on chromosome <mask>", "{chromosome}"),
    ("The start position of {symbol} on chromosome {chromosome} is <mask>", "{start}"),
    ("The starting base pair of {symbol} on chromosome {chromosome} is <mask>", "{start}"),
    ("{symbol} in human is located on chromosome {chromosome}, the start position is about <mask>", "{start}"),
    ("The end position of {symbol} on chromosome {chromosome} is <mask>", "{end}"),
    ("The ending base pair of {symbol} on chromosome {chromosome} is <mask>", "{end}"),
    ("{symbol} in human is located on chromosome {chromosome}, the end position is about <mask>", "{end}")
]

# Generate prompts and responses from the dataset
for _, row in df.iterrows():
    for input_template, response_template in templates:
        prompts.append("You are a bioinformatics expert. Based on the following text, use the correct words to fill out the masked part.")
        inputs.append(input_template.format(symbol=row['Gene Symbol'], chromosome=row['Chromosome']))
        responses.append(response_template.format(gene_id=row['Gene ID'], chromosome=row['Chromosome'], start=row['Start'], end=row['End']))

# Create and save the DataFrame
df_fine_tuning = pd.DataFrame({'Prompt': prompts, 'Input': inputs, 'Response': responses})
df_fine_tuning.to_parquet(output_parquet_file, index=False)

print(f"Parquet file saved to: {output_parquet_file}")
