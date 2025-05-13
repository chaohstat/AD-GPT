import pandas as pd
import os

def read_mim2gene(filename):
    # 读取 mim2gene 文件并创建 Gene Symbol 到 MIM Number 的映射
    gene_to_mim = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            if not line.startswith("#"):
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    mim_number = parts[0]
                    gene_symbol = parts[3]
                    if gene_symbol:
                        gene_to_mim[gene_symbol] = mim_number
    
    # 添加额外的基因映射
    additional_mappings = {"SMS1": "611573", "mTOR": "601231", "C3": "120700"}
    gene_to_mim.update(additional_mappings)
    
    # 检查额外基因是否被包含
    for gene, mim in additional_mappings.items():
        if gene not in gene_to_mim:
            print(f"警告: 基因 {gene} 未成功添加到 gene_to_mim 映射中")
    
    return gene_to_mim


# 读取 144AD 基因列表
ad_genes = pd.read_excel(r"C:\Users\87485\OneDrive - Florida State University\LLM\code\preprocessing\after1031\144AD.xlsx")["Gene symbol"].tolist()

# 读取 mim2gene 文件并生成映射
gene_to_mim = read_mim2gene("mim2gene.txt")

# 保存 Gene Symbol 和 MIM Number 的配对
result = []

# 匹配并保存结果
for gene in ad_genes:
    mim_number = gene_to_mim.get(gene)
    if mim_number:
        result.append([gene, mim_number])
    else:
        print(f"基因 {gene} 未找到对应的 MIM Number")

# 创建 DataFrame 并保存为 CSV 文件
df = pd.DataFrame(result, columns=["Gene symbol", "MIM Number"])
output_path = r"C:\Users\87485\OneDrive - Florida State University\LLM\code\preprocessing\after1031\gene_mim_mapping.csv"
df.to_csv(output_path, index=False)
print(f"CSV 文件已生成：{output_path}")
