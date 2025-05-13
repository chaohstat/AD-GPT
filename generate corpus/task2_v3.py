import pandas as pd 
# Load positive and negative samples
positive_file_path = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\combined.csv'
negative_file_path = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\combine_neg.csv'

positive_df = pd.read_csv(positive_file_path)
negative_df = pd.read_csv(negative_file_path)

# Combine positive and negative samples
positive_df['Response'] = "Yes"
negative_df['Response'] = "No"
positive_df['Region'] = positive_df['Region'].str.replace('Brain_', '', regex=False)
# Concatenate the dataframes
combined_df = pd.concat([positive_df, negative_df], ignore_index=True)

# Initialize result lists
prompts = []
inputs = []
responses = []

# Templates for questions
eQTL_question_templates = [
    "Does the gene {gene} have any SNPs in the {region} region that significantly affect expression regulation?",
    "Are there significant SNPs for {gene} in the {region} region that impact expression?",
    "Can you confirm if {gene} has significant SNPs in the {region} that play a role in expression regulation?",
    "Is there any SNP in the {region} region for {gene} that influences its expression significantly?",
    "Are expression-regulating SNPs present for {gene} in the {region} significantly?"
]

sQTL_question_templates = [
    "Does the gene {gene} have any SNPs in the {region} region that significantly affect splicing regulation?",
    "Are there significant SNPs for {gene} in the {region} region that impact splicing?",
    "Can you confirm if {gene} has significant SNPs in the {region} that play a role in splicing regulation?",
    "Is there any SNP in the {region} region for {gene} that influences its splicing significantly?",
    "Are splicing-regulating SNPs present for {gene} in the {region} significantly?"
]

# Iterate over combined samples
for _, row in combined_df.iterrows():
    gene = row['Gene Symbol']
    region = row['Region']
    response = row['Response']

    # Generate eQTL questions
    for question_template in eQTL_question_templates:
        prompt_exp = "You are the best genetic expert on Alzheimer's disease in the world. Please make a judgment based on the following and carefully check the reliability of your reasoning process and return 'Yes' or 'No' only in your response."
        input_exp = question_template.format(gene=gene, region=region)
        prompts.append(prompt_exp)
        inputs.append(input_exp)
        responses.append(response)

    # Generate sQTL questions
    for question_template in sQTL_question_templates:
        prompt_spl = "You are the best genetic expert on Alzheimer's disease in the world. Please make a judgment based on the following and carefully check the reliability of your reasoning process and return 'Yes' or 'No' only in your response."
        input_exp = question_template.format(gene=gene, region=region)
        prompts.append(prompt_spl)
        inputs.append(input_exp)
        responses.append(response)

# Create DataFrame for fine-tuning
df_fine_tuning = pd.DataFrame({
    'Prompt': prompts,
    'Input': inputs,
    'Response': responses
})
df_fine_tuning = df_fine_tuning.sample(frac=1, random_state=42).reset_index(drop=True)
# Save to parquet file
output_parquet_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\1031\task2_v4.parquet'
df_fine_tuning.to_parquet(output_parquet_file, index=False)

print(f"Parquet file saved to: {output_parquet_file}")
