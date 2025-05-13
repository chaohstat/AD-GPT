import os
import pandas as pd
import gzip

# 读取144AD基因的Excel文件
ad_genes_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\rawdata\144AD.xlsx'
ad_genes_df = pd.read_excel(ad_genes_file, engine='openpyxl')  # 使用 openpyxl 引擎

# 提取基因符号
ad_genes = ad_genes_df['Gene symbol'].tolist()

# 设置处理过的GTEx文件夹路径
folder_path = r'C:\Users\87485\OneDrive - Florida State University\LLM\rawdata\gtexdata\GTEx_Analysis_v8_sQTL'

# 过滤出以 v8.sgenes.txt.gz 结尾的文件
files = [f for f in os.listdir(folder_path) if f.endswith('v8.sgenes.txt.gz')]

# 初始化一个结果列表
result_list = []

# 循环读取文件并匹配AD相关基因
for file in files:
    file_path = os.path.join(folder_path, file)
    
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t')
        
        # 筛选基因符号匹配的基因
        df_filtered = df[df['gene_name'].isin(ad_genes)]
        
        if not df_filtered.empty:
            # 对筛选后的数据进行深拷贝
            df_filtered = df_filtered[['gene_id', 'gene_name', 'gene_chr', 'gene_start', 'gene_end', 'strand', 
                                       'variant_id', 'pval_nominal', 'pval_nominal_threshold']]
            
            # 从文件名中提取区域信息
            base_file_name = file.split('.v8')[0]
            df_filtered.loc[:, 'region'] = base_file_name
            
            # 过滤出影响区域的SNP
            df_significant = df_filtered[df_filtered['pval_nominal'] < df_filtered['pval_nominal_threshold']].copy()

            if not df_significant.empty:
                result_list.append(df_significant)

# 合并所有文件的筛选结果
combined_df = pd.concat(result_list, ignore_index=True)

# 按基因整理影响区域，确保输出正确的Region（SNP和区域信息）
output_list = []
for gene_name, group in combined_df.groupby('gene_name'):
    for _, row in group.iterrows():
        # 只处理以 'Brain' 开头的区域
        if row['region'].startswith('Brain'):
            # 每个基因的多个区域和SNP在新行中重复记录
            output_list.append({
                'Gene ID': row['gene_id'],
                'Gene Symbol': row['gene_name'],
                'Chromosome': row['gene_chr'],
                'Start': row['gene_start'],
                'End': row['gene_end'],
                'Strand': row['strand'],
                'Variant ID': row['variant_id'],
                'P-value Nominal': row['pval_nominal'],
                'P-value Threshold': row['pval_nominal_threshold'],
                'Region': row['region'],
                'Source': 'sQTL'  # 添加 source 列，内容为 'sQTL'
            })

# 将结果保存为CSV文件
output_csv_file = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\AD_related_genes_impact_regions_with_brain_regions.csv'
output_df = pd.DataFrame(output_list)
output_df.to_csv(output_csv_file, index=False)

print(f"结果已保存到 {output_csv_file}")
