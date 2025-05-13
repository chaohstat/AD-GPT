import pandas as pd
from pathlib import Path
from nlp_process import basic_clean_pipe

# 定义文件路径
combined_path = r"C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\combined.csv"
gene_mapping_path = r"C:\Users\87485\OneDrive - Florida State University\LLM\code\preprocessing\after1031\gene_mim_mapping.csv"
molecular_genetics_dir = Path(r"C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\omim mole")
output_parquet_path_train = r"C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\1031\task4_train_v2.parquet"
output_parquet_path_test = r"C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\1031\task4_test_v2.parquet"

# 函数用于读取 molecular genetics 文件内容
def get_molecular_genetics(mim_number):
    file_path = molecular_genetics_dir / f"{mim_number}_molecularGenetic.txt"
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    return None  # 返回 None 表示文件不存在

# 提取与阿尔茨海默病相关的内容
def extract_alzheimers_related_content(text):
    keywords = ["Alzheimer's disease", "Alzheimer Disease", "Alzheimer disease"]
    paragraphs = text.split('\n\n')
    
    # Extract paragraphs containing any keyword
    related_paragraphs = [
        paragraph for paragraph in paragraphs if any(keyword in paragraph for keyword in keywords)
    ]
    
    # Combine the extracted paragraphs if any are found
    if related_paragraphs:
        combined_text = '\n\n'.join(related_paragraphs)
        combined_text = combined_text.replace("AD", "Alzheimer disease")
        return basic_clean_pipe(combined_text)
    return basic_clean_pipe(text)

# 加载数据文件
combined_df = pd.read_csv(combined_path)
gene_mim_mapping_df = pd.read_csv(gene_mapping_path)

# 提取所有大脑区域
brain_regions = combined_df['Region'].unique()

# 提取存在分子遗传学文件的基因 MIM IDs，并查找对应的基因符号
genes_with_files_mim = {int(file.stem.split('_')[0]) for file in molecular_genetics_dir.glob("*_molecularGenetic.txt")}
genes_with_files = gene_mim_mapping_df[gene_mim_mapping_df['MIM Number'].isin(genes_with_files_mim)]

# 准备新的数据集
data = []
for brain_region in brain_regions:
    for _, gene_row in genes_with_files.iterrows():
        gene = gene_row['Gene symbol']
        mim_number = gene_row['MIM Number']
        molecular_genetics_summary = get_molecular_genetics(mim_number)
        
        # 仅处理存在分子遗传学文件的基因
        if molecular_genetics_summary is None:
            continue
        molecular_genetics_summary = extract_alzheimers_related_content(molecular_genetics_summary)

        # 检查该基因是否与特定大脑区域有关系
        region_gene_match = combined_df[
            (combined_df['Region'] == brain_region) & 
            (combined_df['Gene Symbol'] == gene) &
            (combined_df['Source'].isin(['eQTL', 'sQTL']))
        ]

        R1 = not region_gene_match.empty
        R2 = gene_row['R2mole'] == 1
        has_relationship = R1 and R2
        # print(brain_region)
        # print(gene)
        # print(f"relationship is {has_relationship}")

        prompt_templates = [
            f"Determine if {brain_region} mediates the relationship between gene {gene} and Alzheimer disease.",
            f"Evaluate the criteria for mediation effect of {brain_region} on the relationship between gene {gene} and Alzheimer disease.",
            f"Determine if {brain_region} has a mediation effect between gene {gene} and Alzheimer disease based on the following criteria."
        ]

        condition=" You need to condiser the following two conditions of genes"

        input_templates = [
            f"1. The gene {gene} {'has' if R1 else 'does not have'} SNPs that affect expression or splicing in {brain_region}. 2. Based on molecular genetics of gene {gene}, there is a relationship between this gene and Alzheimer’s disease. The molecular genetics summary is {molecular_genetics_summary}",
            f"1. There {'are' if R1 else 'are no'} significant SNPs on {gene} which affect expression or splicing in {brain_region}. 2. Based on the molecular genetics of gene {gene}: {molecular_genetics_summary}, you can find the potential relationship between {gene} and Alzheimer disease",
            f"1. {gene} {'has' if R1 else 'does not have'} significant SNPs in the {brain_region} that play a role in splicing or expression regulation. 2. The molecular genetics summary of {gene} is: {molecular_genetics_summary}. And there shows a realtionship between {gene} and Alzheimer disease"
        ]

        output_text = (
            f"{brain_region} has mediation effect between gene {gene} and Alzheimer disease"
            if has_relationship else
            f"{brain_region} has no mediation effect between gene {gene} and Alzheimer disease"
        )

        # 遍历不同的模板组合来生成数据
        for prompt in prompt_templates:
            for input_text in input_templates:
                data.append({
                    "Prompt": prompt.strip()+condition,
                    "Input": input_text.strip(),
                    "Response": output_text.strip()
                })

# 转换为 DataFrame
relationship_df = pd.DataFrame(data)

# 打乱数据集
relationship_df = relationship_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 将数据集分割成训练集和测试集
train_size = int(len(relationship_df) * 2 / 3)
train_df = relationship_df.iloc[:train_size]
test_df = relationship_df.iloc[train_size:]

# 保存为 Parquet 文件
train_df.to_parquet(output_parquet_path_train, index=False)
test_df.to_parquet(output_parquet_path_test, index=False)
print(f"Training data saved to {output_parquet_path_train}")
print(f"Test data saved to {output_parquet_path_test}")
