import pandas as pd
from pathlib import Path
from nlp_process import basic_clean_pipe

# 定义文件路径
gene_mapping_path = r"C:\Users\87485\OneDrive - Florida State University\LLM\code\preprocessing\after1031\gene_mim_mapping.csv"
molecular_genetics_dir = Path(r"C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\omim mole")
output_train_parquet_path = r"C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\1031\task3mole_train_v3.parquet"
output_test_parquet_path = r"C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\1031\task3mole_test_v3.parquet"

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

# 生成 prompt, input, output 格式的数据
data = []
for _, row in gene_mim_mapping_df.iterrows():
    gene_symbol = row["Gene symbol"]
    mim_number = row["MIM Number"]
    relationship = row["R2mole"]

    # 仅在分子遗传学文件存在的情况下处理
    molecular_genetics_summary = get_molecular_genetics(mim_number)
    if molecular_genetics_summary is None:
        continue  # 跳过没有分子遗传学文件的基因
    molecular_genetics_summary=extract_alzheimers_related_content(molecular_genetics_summary)
    # 对分子遗传学摘要进行清理
    molecular_genetics_summary = basic_clean_pipe(molecular_genetics_summary)

    # 多种 prompt 和 input 的表达方式
    # Begin = "Answer *Yes* or *No* only in your response. "
    prompt_templates = [
        f"Confirm the relationship between {gene_symbol} and Alzheimer Disease based on the molecular genetics summary.",
        f"Please evaluate if there is a relationship between {gene_symbol} and Alzheimer disease in light of its molecular genetics.",
        f"Determine if {gene_symbol} has a role in Alzheimer disease based on the molecular genetics summary.",
        f"Assess whether {gene_symbol} plays a role in Alzheimer disease according to the molecular genetics summary."
    ]

    instruction_templates = [
        f"Instruction: You need to use determined formats in your response. Please respond with *There is a relationship between {gene_symbol} and Alzheimer disease based on the molecular genetics summary* if there is a relationshipor or *The relationship between {gene_symbol} and Alzheimer disease is not clear based on the molecular genetics summary.* if the relationship is unclear",
        f"Instruction: Use the following formats to answer the question: - If there is a relationship: *There is a relationship between {gene_symbol} and Alzheimer disease based on the molecular genetics summary.* - If the relationship is unclear: *The relationship between {gene_symbol} and Alzheimer disease is not clear based on the molecular genetics summary.*"
    ]


    input_templates = [
        f"The molecular genetics summary of {gene_symbol} is: {molecular_genetics_summary}",
        f"Here is the molecular genetics summary for {gene_symbol}: {molecular_genetics_summary}",
        f"The Omim website shows the molecular genetics summary of {gene_symbol} is: {molecular_genetics_summary}",
        f"{gene_symbol} has the following molecular genetics summary: {molecular_genetics_summary} ",
        f"The molecular genetic summary for {gene_symbol}: "

    ]

    output_text = (
        f"There is a relationship between {gene_symbol} and Alzheimer disease based on the molecular genetics summary"
        if relationship == 1
        else f"The relationship between {gene_symbol} and Alzheimer disease is not clear based on the molecular genetics summary."
    )

    for prompt in prompt_templates:
        for input_text in input_templates:
            for instruction in instruction_templates:
                data.append({
                    "Prompt":prompt + " " + instruction,
                    "Input": input_text,
                    'Response': output_text
                })

# 转换为 DataFrame
relationship_df = pd.DataFrame(data)

# 打乱数据集
relationship_df = relationship_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 将数据集拆分为训练集和测试集
train_size = int(len(relationship_df) * 4 / 5)
train_df = relationship_df.iloc[:train_size]
test_df = relationship_df.iloc[train_size:]

# 保存为 Parquet 文件
train_df.to_parquet(output_train_parquet_path, index=False)
test_df.to_parquet(output_test_parquet_path, index=False)
print(f"Training data saved to {output_train_parquet_path}")
print(f"Test data saved to {output_test_parquet_path}")
