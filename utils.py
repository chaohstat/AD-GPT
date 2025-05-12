import re
import torch
import pandas as pd
def extract_masked_words(masked_sentence, unmasked_sentence, mask_token='<mask>'):
    masked_words = []
    pattern = r'\s+|[,]+|(?<!\w)\.(?!\w)|(?<!c)-'
    masked_tokens = re.split(pattern,masked_sentence)
    masked_tokens = [tokens for tokens in masked_tokens if tokens]
    unmasked_tokens = re.split(pattern,unmasked_sentence)
    unmasked_tokens = [tokens for tokens in unmasked_tokens if tokens]
    i = 0  # Index for masked_tokens
    j = 0  # Index for unmasked_tokens
    start_token = ""
    next_token = ""
    mask_count = 0
    while i < len(masked_tokens):
        if masked_tokens[i] != mask_token:
            # If current token is not a mask, move to the next token in both sentences
            if start_token and mask_count>0:
                next_token = masked_tokens[i]
                if start_token and next_token:
                    add_flag = False
                    if start_token == '0':
                        add_flag = True
                    for j in range(j,len(unmasked_tokens)):
                        if unmasked_tokens[j]==next_token:
                            mask_count=0
                            next_token = ""
                            break
                        if add_flag == True:
                            masked_words.append(unmasked_tokens[j])
                        if unmasked_tokens[j]==start_token:
                            add_flag = True
                    
            start_token = masked_tokens[i]
        else:
            mask_count +=1
            if i == 0:
                add_flag = True
                start_token = '0'
            if i == len(masked_tokens)-1:
                add_flag = False
                for j in range(j,len(unmasked_tokens)):
                        if add_flag == True:
                            masked_words.append(unmasked_tokens[j])
                        if unmasked_tokens[j]==start_token:
                            add_flag = True
        i+=1
        
    return masked_words

def extract_response(text):
    import re
    # Define the pattern to search for
    pattern = r'### Response:\n(.*?)</s>'
    # Use re.search to find the pattern in the text
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Extract the group that contains the response
        response = match.group(1).strip()
        return response
    else:
        return None

# def extract_response_llama3(text):
#     import re
#     # Define the pattern to search for
#     pattern = r'### Response:\n(.*?)<\|end_of_text\|>'
#     # Use re.search to find the pattern in the text
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         # Extract the group that contains the response
#         response = match.group(1).strip()
#         return response
#     else:
#         return None
# def extract_response_llama3(text):
#     """
#     Extract the last response following '### Response:' and ending before '<|eot_id|>'.
#     """
#     # Define the pattern to match all responses
#     pattern = r'### Response:\n(.*?)(?=(### Response:|<\|eot_id\|>))'
#     # Use re.findall to get all matches
#     matches = re.findall(pattern, text, re.DOTALL)
    
#     if matches:
#         # Take the last match and extract the actual response
#         last_response = matches[-1][0].strip()
#         return last_response
#     else:
#         return None    
def extract_response_llama3(text):
    """
    Extract the last response in the Llama3 instruct format.
    The response starts with '<|start_header_id|>assistant<|end_header_id|>' 
    and ends with '<|eot_id|>'.
    """
    # Define the pattern to match all responses in the Llama3 instruct format
    pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n(.*?)(?=(<\|start_header_id\|>|<\|eot_id\|>))'
    
    # Use re.findall to get all matches
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Take the last match and extract the actual response
        last_response = matches[-1][0].strip()
        return last_response
    else:
        return None
#%%

def create_prompt_mask(sample):
    bos_token = '<s>'
    original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    # system_message = "you are bioinformatic expert, based on text, use correct words to fill out the masked part"
    system_message = sample['Prompt']
    input= sample['Input'].replace(original_system_message, "").replace(system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
    response = sample["Response"]
    eos_token = '</s>'
    input = input.replace('<mask>', tokenizer.mask_token)
    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input:"
    full_prompt += "\n" + input
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + response
    full_prompt += eos_token
    return full_prompt
def create_prompt(sample):
    bos_token = '<s>'
    original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    # system_message = "you are bioinformatic expert, based on text, use correct words to fill out the masked part"
    system_message = sample['Prompt']
    input= sample['Input'].replace(original_system_message, "").replace(system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
    response = sample["Response"]
    eos_token = '</s>'
    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input:"
    full_prompt += "\n" + input
    full_prompt += "\n\n### Response:"
    return full_prompt    

def create_prompt_out(sample):
    bos_token = '<s>'
    original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    system_message =sample['Prompt']
    input= sample['Input'].replace(original_system_message, "").replace(system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
    # input = input.replace('<mask>', tokenizer.mask_token)
    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Instruction:\n"
    full_prompt += system_message
    full_prompt += "\n\n### Input:\n"
    full_prompt += input
    full_prompt += "\n\n### Response:\n"
    return full_prompt

def compute_metrics_task2(valid_data,tokenizer, model):
    total_correct = 0
    fn= 0
    fp = 0
    total_check = 0
    label_map = {
    0:'No',
    1:'Yes'
    }
    for i, row in valid_data.iterrows():
        # prompt = create_prompt_out(row)
        prompt = create_test_prompt(row['Input'],task=2)
        
        predicted_class = generate_response_bert(prompt,tokenizer,model)
        pred_words = label_map[predicted_class]
        true_words = row['Response']
        # Extract masked words from the prompt and responses
        # true_words = extract_masked_words(row['Prompt'], row['Response'])
        # pred_words = extract_masked_words(row['Prompt'], pred_sentence)
        
        # Update total words count
        total_check += 1
        
        # Compare true and predicted words
        if pred_words==true_words:
            total_correct+=1
        elif pred_words=="Yes" and true_words=="No":
            fp+=1
        elif pred_words=="No" and true_words=="Yes":
            fn+=1
        print('Pred:',pred_words)   
        print('True:',true_words)
    # Calculate metrics
    
    # Precision and Recall are equal in this case
    precision = total_correct / (total_correct + fp) if (total_correct + fp) > 0 else 0
    recall = total_correct / (total_correct + fn) if (total_correct + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics
    print(f'Accuracy: {precision:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

    return {'accuracy': precision, 'precision': precision, 'recall': recall, 'f1_score': f1_score}
import re

def extract_yes_no(response: str) -> str:
    """
    Extracts 'Yes' or 'No' from the model response while ignoring any additional text.
    It looks for the first occurrence of 'Yes' or 'No' after '### Response:'.
    """
    # Ensure response starts at the expected section
    match = re.search(r'### Response:\s*(Yes|No|None)\b', response, re.IGNORECASE)
    
    if match:
        if match == "None":
            match = 'No'
        return match.group(1).capitalize()  # Ensure consistent capitalization of 'Yes' or 'No'
    
    return "Unknown"  # Default return if no valid match is found

def compute_metrics_llama3(valid_data,tokenizer, model):
    total_correct = 0
    total_check = 0
    fp=fn=0
    data_prompt = """
        You are the best genetic expert on Alzheimer's disease in the world. Answer **strictly with 'Yes' or 'No'** and **do not provide any explanation**.

        ### Instruction:
        {}

        ### Response:
        """
    for i, row in valid_data.iterrows():
        # prompt = create_prompt_out(row)
        prompt = data_prompt.format(row["Input"])
        input_ids = tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda")
        outputs = model.generate(
            **input_ids,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id
        )
        pred_words = extract_yes_no(tokenizer.batch_decode(outputs)[0])
        true_words = row['Response']
        # Extract masked words from the prompt and responses
        # true_words = extract_masked_words(row['Prompt'], row['Response'])
        # pred_words = extract_masked_words(row['Prompt'], pred_sentence)
        
        # Update total words count
        total_check += 1
        if pred_words == "None": pred_words="No"
        # Compare true and predicted words
        if pred_words==true_words:
            total_correct+=1
        elif pred_words=="Yes" and true_words=="No":
            fp+=1
        elif pred_words=="No" and true_words=="Yes":
            fn+=1
        # Compare true and predicted words
        if pred_words==true_words:
            total_correct+=1
        print('Pred:',pred_words)   
        print('True:',true_words)

    # Precision and Recall are equal in this case
    precision = total_correct / (total_correct + fp) if (total_correct + fp) > 0 else 0
    recall = total_correct / (total_correct + fn) if (total_correct + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics
    print(f'Accuracy: {precision:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

    return {'accuracy': precision, 'precision': precision, 'recall': recall, 'f1_score': f1_score}
def generate_response(prompt, tokenizer, model):
  encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')
  
  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0]

def generate_response_bert(prompt,tokenizer,model):
    # Tokenize
    MAX_LEN=128
    encoded_input = tokenizer(
        prompt,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    encoded_input = encoded_input.to(model.device)
    # Move tensors to device
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    token_type_ids = encoded_input.get('token_type_ids', None)
    if token_type_ids is not None and token_type_ids.nelement() > 0:
        token_type_ids = token_type_ids
    else:
        token_type_ids = None
    # Run inference
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class
def extract_identifier(text):
    # Adjusted regex to capture content after <extra_id_0> and stop at </s> if present
    match = re.search(r"<extra_id_0>\s*([^<]+?)(?=</s>|$)", text)
    return match.group(1).strip() if match else None
def metrics_for_t5_mask(data,tokenizer,model):
    accuracy = 0 
    for i, row in data.iterrows():
        prompt = row['input_ids']
        output = generate_response(prompt,tokenizer,model)
        id1 = extract_identifier(output)
        id2 = extract_identifier(row['labels'])
        if id1 == id2:
            accuracy+=1
    accuracy = accuracy/len(data)
    return accuracy

def create_test_prompt(text,task):
    bos_token = '<s>'
    full_prompt = ''
    if task==1:
        system_message = 'You are a bioinformatics expert. Answer the following question accurately and professionally.'
        full_prompt += bos_token
        full_prompt += "### Instruction:"
        full_prompt += "\n" + system_message
        full_prompt += "\n\n### Input:"
        full_prompt += "\n" + text
        full_prompt += "\n\n### Response:\n"
    elif task == 2:
        system_message = "You are the best genetic expert on Alzheimer's disease in the world. Please make a judgment based on the following and carefully check the reliability of your reasoning process and return 'Yes' or 'No' only in your response."
        full_prompt += bos_token
        full_prompt += "### Instruction:"
        full_prompt += "\n" + system_message
        full_prompt += "\n\n### Input:"
        full_prompt += "\n" + text
        full_prompt += "\n\n### Response:\n"
    return full_prompt

def extract_gene_names_regex(query):
    # Regex pattern: Start with uppercase letters, may include numbers, typically 2-5 characters
    pattern = r'\b[A-Z0-9]{2,}\b'
    return re.findall(pattern, query)        

data_prompt = """You are a bioinformatics expert. Answer the following question accurately and professionally.

### Input:
{}

### Response:
{}"""

def extract_assistant_response(text):
    """
    Extracts the assistant's response from the given text.
    
    Parameters:
        text (str): The input text containing the assistant's response.
    
    Returns:
        str: The extracted assistant response.
    """
    match = re.search(r'<\|Assistant\|><think>.*?</think>(.*?)<\|end▁of▁sentence\|>', text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return "No assistant response found."
from unsloth import to_sharegpt,standardize_sharegpt,apply_chat_template
from datasets import Dataset
from unsloth.chat_templates import get_chat_template
def data_generation(json_path,tokenizer,chat_model):
    dataset = Dataset.from_json(json_path)
    if chat_model=="llama3_instruct":
        tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )
    elif chat_model=="Qwen":
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "alpaca", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
            mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
        )
    dataset = dataset.rename_columns({"question":"instruction","sql":"output"})
    dataset = to_sharegpt(
        dataset,
        merged_prompt="{instruction}",
        merged_column_name="instruction",
        output_column_name="output",
        conversation_extension=3
    )
    dataset = standardize_sharegpt(dataset)
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    pass
    dataset = dataset.map(formatting_prompts_func,batched=True)
    return dataset
# %%
