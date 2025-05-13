import pandas as pd

# Load the original dataset
file_path = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\combined.csv'
df = pd.read_csv(file_path)

# Predefined region names
all_regions = [
    'Heart_Atrial_Appendage',
    'Heart_Left_Ventricle',
    'Kidney_Cortex',
    'Liver',
    'Lung',
    'Minor_Salivary_Gland',
    'Muscle_Skeletal',
    'Nerve_Tibial',
    'Prostate',
    'Skin_Not_Sun_Exposed_Suprapubic',
    'Skin_Sun_Exposed_Lower_leg',
    'Small_Intestine_Terminal_Ileum',
    'Spleen',
    'Stomach'
]

# Create a new dataframe with negative samples
combine_neg_df = df.copy()
combine_neg_df['Region'] = all_regions * (len(combine_neg_df) // len(all_regions)) + all_regions[:len(combine_neg_df) % len(all_regions)]

# Save the new negative samples to a CSV file
output_csv_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\combine_neg.csv'
combine_neg_df.to_csv(output_csv_file, index=False)

print(f"Negative samples saved to: {output_csv_file}")
