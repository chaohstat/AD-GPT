import pandas as pd
from pathlib import Path
from nlp_process import basic_clean_pipe

# 定义文件路径
gene_mapping_path = r"C:\Users\87485\OneDrive - Florida State University\LLM\code\preprocessing\after1031\gene_mim_mapping.csv"
molecular_genetics_dir = Path(r"C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\omim mole")
output_parquet_path = r"C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\1031\task3gene_molecular_genetics.parquet"

# 读取 gene_mim_mapping 文件
gene_mim_mapping_df = pd.read_csv(gene_mapping_path)

# 函数用于读取 molecular genetics 文件内容
def get_molecular_genetics(mim_number):
    file_path = molecular_genetics_dir / f"{mim_number}_molecularGenetic.txt"
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    return None  # 返回 None 表示文件不存在

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

# 提取基因与分子遗传学摘要的对应关系
gene_molecular_data = []
for _, row in gene_mim_mapping_df.iterrows():
    gene_symbol = row["Gene symbol"]
    mim_number = row["MIM Number"]

    # 获取分子遗传学摘要
    molecular_genetics_summary = get_molecular_genetics(mim_number)
    molecular_genetics_summary=extract_alzheimers_related_content(molecular_genetics_summary)

    if molecular_genetics_summary is not None:  # 仅保留存在摘要的基因
        gene_molecular_data.append({
            "Gene symbol": gene_symbol,
            #"MIM Number": mim_number,
            "Molecular Genetics Summary": molecular_genetics_summary
        })

# 转换为 DataFrame
gene_molecular_df = pd.DataFrame(gene_molecular_data)

# 保存为 Parquet 文件
gene_molecular_df.to_parquet(output_parquet_path, index=False)
print(f"Gene and molecular genetics data saved to {output_parquet_path}")
