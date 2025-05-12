import os
import numpy as np
import pandas as pd
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported
import sqlite3
import random
# Saving model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Warnings
import warnings
warnings.filterwarnings("ignore")

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    device_map="auto",
    dtype=None
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=False,
    bias = 'none',
    use_gradient_checkpointing="unsloth",
    random_state = 32,
    loftq_config = None,
)

# data_prompt = """Given the genetics summary for the gene {}, tell me the relation between it and Alzheimer's Disease and give me the reasoning.

# ### Input:
# {}

# ### Response:
# {}"""

# Connect to the SQLite database
conn = sqlite3.connect("gene_database_v2.db")
cursor = conn.cursor()

# Query data from the database
cursor.execute("""
SELECT genes.gene_name, summaries.summary, summaries.relation, summaries.reasoning 
FROM genes
JOIN summaries ON genes.id = summaries.gene_id
""")
rows = cursor.fetchall()

# Define the end-of-sequence token
EOS_TOKEN = tokenizer.eos_token
chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.
### Instruction:
{INPUT}

### Response:
{OUTPUT}"""
# Format prompts using the database data
# def formatting_prompt(rows):
    
#     for gene_name, summary, relation, reasoning in rows:
#         texts= []
#         choice = random.choice([0,1])
#         if choice==0:
#             relation_text = f"There is potential relation between{gene_name} and Alzheimer's Disease " if relation else f"There is no obvious evidence to support the relation between{gene_name} and Alzheimer's Disease"
#             response = f"{relation_text}. Reasoning: {reasoning}"
#             text = data_prompt.format(gene_name, summary, response) + EOS_TOKEN
            
#             texts.append({"content":text,"role":'user'})
#     return texts

user_input_format1 = "tell me the relation between the gene {} and Alzheimer's Disease and give me the reasoning."
user_input_format2 = "give me the relation between the gene {} and Alzheimer's Disease based on the summary"
variations_format1 = [
    "Can you explain the connection between the gene {} and Alzheimer's Disease, and provide your reasoning?",
    "What is the relationship between the gene {} and Alzheimer's Disease? Please explain your reasoning.",
    "I'd like to know the relation between the gene {} and Alzheimer's Disease. Could you provide details and reasoning?",
    "Tell me if there’s a relationship between the gene {} and Alzheimer's Disease, and explain why.",
    "Explain how the gene {} relates to Alzheimer's Disease, and give a detailed reasoning.",
    "What connection, if any, exists between the gene {} and Alzheimer's Disease? Please justify your answer.",
    "Could you describe the relation between {} and Alzheimer's Disease and the reasoning behind it?",
    "I want to understand the relation between the gene {} and Alzheimer's Disease. What’s the reasoning for it?",
    "Is there a relation between the gene {} and Alzheimer's Disease? Explain why or why not.",
    "Provide an explanation of the relationship between the gene {} and Alzheimer's Disease, and include the reasoning."
]

# Variations for Format 2
variations_format2 = [
    "Based on the summary, what is the relation between the gene {} and Alzheimer's Disease?",
    "Can you tell me the connection between the gene {} and Alzheimer's Disease according to the provided summary?",
    "From the given summary, explain the relationship between the gene {} and Alzheimer's Disease.",
    "Using the summary, what is the relation between {} and Alzheimer's Disease?",
    "What does the summary say about the relation between the gene {} and Alzheimer's Disease?",
    "According to the summary, is there a relation between the gene {} and Alzheimer's Disease? Please explain.",
    "Based on the given summary, explain the relation between {} and Alzheimer's Disease.",
    "Tell me about the relationship between the gene {} and Alzheimer's Disease using the provided summary.",
    "What is the relationship between {} and Alzheimer's Disease as described in the summary?",
    "Can you analyze the summary and describe the connection between the gene {} and Alzheimer's Disease?"
]
variations_format3 = [
    "Tell me more about this gene's possible relationship with Alzheimer's Disease",
    "Can you explain this gene's connection to Alzheimer's Disease?",
    "What is the role of this gene in Alzheimer's Disease?",
    "How is this gene related to Alzheimer's Disease?",
    "Is there evidence linking this gene to Alzheimer's Disease?",
    "What is known about this gene's involvement in Alzheimer's Disease?",
    "Does this gene play a role in Alzheimer's Disease development?",
    "Provide details on this gene's potential link to Alzheimer's Disease.",
    "Can you elaborate on this gene's association with Alzheimer's Disease?",
    "What research supports this gene's connection to Alzheimer's Disease?",
    "Is there any evidence that this gene affects Alzheimer's Disease?",
    "How does this gene contribute to Alzheimer's Disease pathology?",
    "What findings suggest a relationship between this gene and Alzheimer's Disease?",
    "What role does this gene have in Alzheimer's Disease progression?",
    "Is this gene considered a risk factor for Alzheimer's Disease?",
    "What insights are available about this gene and Alzheimer's Disease?",
    "How strong is the association between this gene and Alzheimer's Disease?",
    "What does research say about this gene's possible role in Alzheimer's Disease?",
    "Does this gene have any known effect on Alzheimer's Disease?",
    "Can you summarize this gene's relevance to Alzheimer's Disease?",
]
# Close the database connection
conn.close()
def formatting_prompt(rows,turns_per_conversation=2):
    def get_related_summary(gene_name):
        conn = sqlite3.connect("gene_database_v2.db")
        cursor = conn.cursor()
        query = """
            SELECT summary, reasoning
            FROM summaries
            JOIN genes ON summaries.gene_id = genes.id
            WHERE genes.gene_name = ? AND summaries.relation = 1
        """
        cursor.execute(query, (gene_name,))
        result = cursor.fetchall()
        conn.close()
        return result
    conversations = []
    task3_questions = []
    for i in range(0,len(rows),turns_per_conversation):
        selected_rows = rows[i:i+turns_per_conversation]
        for gene_name, summary, relation, reasoning in selected_rows:
            conversation= []
            choice = random.choice([0,1])
            if choice==0:
                relation_text = f"There is potential relation between{gene_name} and Alzheimer's Disease " if relation else f"There is no obvious evidence to support the relation between{gene_name} and Alzheimer's Disease"
                response = f"{relation_text}. According to:\n {summary}\n Reasoning: {reasoning}"
                user_input_format1 = random.sample(variations_format1,1)[0]
                user_input = user_input_format1.format(gene_name) 
                conversation.append({"content":user_input,"role":'user'})
                task3_questions.append(user_input)
                conversation.append({'content':response,'role':'assistant'})
            if choice==1:
                user_input_format2 = random.sample(variations_format2,1)[0]
                user_input=user_input_format2.format(gene_name)
                conversation.append({"content":user_input,"role":'user'})
                task3_questions.append(user_input)
                conversation.append({"content":"Please give your summary","role":'assistant'})
                conversation.append({"content":summary,"role":"user"})
                relation_text = f"There is potential relation between{gene_name} and Alzheimer's Disease " if relation else f"There is no obvious evidence to support the relation between{gene_name} and Alzheimer's Disease"
                response = f"{relation_text}.\nReasoning: {reasoning}"
                conversation.append({"content":response,"role":"assistant"})
                if relation == False:
                    user_input = random.sample(variations_format3, 1)[0]
                    conversation.append({"content": user_input, "role": "user"})
                    task3_questions.append(user_input)
                    # Search the database for summaries with a relation flag of True
                    related_summaries = get_related_summary(gene_name)
                    if related_summaries:
                        related_response = "Here is a summary from the database indicating a relation:\n"
                        for rel_summary, rel_reasoning in related_summaries:
                            related_response += f"- Summary: {rel_summary}\n  Reasoning: {rel_reasoning}\n"
                        conversation.append({"content": related_response.strip(), "role": "assistant"})
                    else:
                        conversation.append({"content": "No related summaries with a positive relation were found in the database.", "role": "assistant"})

        conversations.append({"conversations":conversation})
    return conversations,task3_questions
# Generate formatted prompts
formatted_data,questions = formatting_prompt(rows)



# Example: print the first few prompts
for prompt in formatted_data[:3]:
    print(prompt)
    print("-" * 80)
dataset= Dataset.from_list(formatted_data)

from unsloth import apply_chat_template
dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)
train_test= dataset.train_test_split(test_size=0.2)
training_data=train_test['train']
testing_data = train_test['test']
trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        num_train_epochs=20,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=5,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()
model.save_pretrained("/nas1/8B_finetuned_llama3.1_task3_chat_v3")
tokenizer.save_pretrained("/nas1/8B_finetuned_llama3.1_task3_chat_v3")
model =FastLanguageModel.for_inference(model)
prompt =  "Given the genetics summary for the gene CHAT, tell me the relation between it and Alzheimer's Disease and give me the reasoning.\n\n### Input:\n{5:Harold et al. (2003)} stated that there was substantial evidence for a susceptibility gene for late-onset Alzheimer disease (AD) on chromosome 10. One of the characteristic features of AD is the degeneration and dysfunction of the cholinergic system. The CHAT gene maps to the linked region of chromosome 10 and was therefore considered both a positional and a functional candidate gene for late-onset AD. {5:Harold et al. (2003)} screened for variants of the CHAT gene in patients with AD and found that none of the 14 variants they identified showed association with AD.\n\n### Response:\n"
inputs = tokenizer(prompt,return_tensors='pt').to('cuda')
outputs = model.generate(**inputs,max_new_tokens=256,use_cache=True)
tokenizer.batch_decode(outputs)

#%%
# BLEU evaluation
from nltk.translate.bleu_score import sentence_bleu
# from utils import extract_response_llama3
bleu_scores = []
questions = []
responses = []

for conversation in testing_data:
    context = []
    for turn in conversation['conversations']:
        if turn['role'] == 'user':
            # Add user input to context
            context.append({"role": "user", "content": turn['content']})
            questions.append(turn['content'])
        elif turn['role'] == 'assistant':
            # Get reference response
            reference_response = turn['content']
            responses.append(turn['content'])
            # Generate model response
            input_ids = tokenizer.apply_chat_template(
                    context,
                    add_generation_prompt=True,  # Add assistant generation token
                    return_tensors="pt"
                )           
            outputs = model.generate(
                    input_ids,
                    max_new_tokens=500,
                    pad_token_id=tokenizer.eos_token_id
                )
            # generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_response = extract_response_llama3(tokenizer.batch_decode(outputs)[0])
            context.append({"role": "assistant", "content": generated_response})
            # Tokenize responses for BLEU
            reference_tokens = [reference_response.split()]  # Wrap in a list for BLEU
            generated_tokens = generated_response.split()

            # Compute BLEU score
            bleu_score = sentence_bleu(reference_tokens, generated_tokens)
            bleu_scores.append(bleu_score)

            # print(f"Context:\n{context}")
            print(f"Reference Response: {reference_response}")
            print(f"Generated Response: {generated_response}")
            print(f"BLEU Score: {bleu_score:.4f}\n")

            # Add the assistant's response to the context
           

# Average BLEU score across all assistant turns
average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu:.4f}")
# %%
