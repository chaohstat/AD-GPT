import pandas as pd
import requests
import os

# 加载 MIM 编号数据，使用你提供的文件路径
file_path = r'C:\Users\87485\OneDrive - Florida State University\LLM\code\preprocessing\after1031\gene_mim_mapping.csv'
gene_mim_data = pd.read_csv(file_path)

# 定义 API URL 和你的 API 密钥
api_key = "T31qrgsyRkKFAgBN9nWtVg"
url = "https://api.omim.org/api/entry"

# 设置输出文件夹路径
output_folder = r'C:\Users\87485\OneDrive - Florida State University\LLM\preprocessed data\omim mole'
os.makedirs(output_folder, exist_ok=True)

# 遍历每个 MIM Number，提取 molecularGenetics 部分
for mim_id in gene_mim_data['MIM Number']:
    # 设置请求参数
    params = {
        "mimNumber": str(mim_id),
        "include": "text",
        "format": "json",
        "apiKey": api_key
    }
    
    try:
        # 发起 API 请求
        response = requests.get(url, params=params)
        
        # 检查请求是否成功
        if response.status_code == 200:
            data = response.json()
            
            # 查找 "molecularGenetics" 部分
            molecular_genetics_content = None
            for text_section in data['omim']['entryList'][0]['entry']['textSectionList']:
                if text_section['textSection'].get('textSectionName') == 'molecularGenetics':
                    molecular_genetics_content = text_section['textSection'].get('textSectionContent')
                    break
            
            # 保存内容到文件
            if molecular_genetics_content:
                file_name = os.path.join(output_folder, f"{mim_id}_molecularGenetic.txt")
                with open(file_name, 'w', encoding='utf-8') as file:
                    file.write(molecular_genetics_content)
                print(f"Saved molecular genetics data for MIM ID {mim_id}.")
            else:
                print(f"No molecular genetics data found for MIM ID {mim_id}.")
        else:
            print(f"Failed to retrieve data for MIM ID {mim_id}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error retrieving data for MIM ID {mim_id}: {e}")
