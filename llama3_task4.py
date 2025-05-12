#%%
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
#%%
# data_prompt = """Given the genetics summary for the gene {}, tell me the relation between it and Alzheimer's Disease and give me the reasoning.

# ### Input:
# {}

# ### Response:
# {}"""

# for conversation in testing_data:
#     context = []
#     for turn in conversation['conversations']:
#         if turn['role'] == 'user':
#             # Add user input to context
#             context.append({"role": "user", "content": turn['content']})
#             questions.append(turn['content'])
#         elif turn['role'] == 'assistant':
#             # Get reference response
#             reference_response = turn['content']
#             responses.append(turn['content'])
# dict = {'instruction':questions,'responses':responses}
# df = pd.DataFrame.from_dict(dict)
# df.to_parquet('task34_test.parquet')
#%%
from datasets import load_from_disk
dataset = load_from_disk("task34_dataset_V3")
#%%
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
        num_train_epochs=3,
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
model.save_pretrained("/nas1/8B_finetuned_llama3.1_task34_chat_v6")
tokenizer.save_pretrained("/nas1/8B_finetuned_llama3.1_task34_chat_v6")
#  # %%
# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import AutoTokenizer, TextStreamer
# from unsloth import FastLanguageModel
# import torch
# import re
# import pandas as pd
# # Load tokenizer and model

# model,tokenizer = FastLanguageModel.from_pretrained(
#     model_name="/nas1/8B_finetuned_llama3.1_task34_chat_v4",
#     load_in_4bit=True,
#     device_map="auto"
# )
# classify_model
# model =FastLanguageModel.for_inference(model)
# df = pd.read_csv("Generated_Prompts_task4.csv")
# from utils import extract_response_llama3
# output = []
# for _,rows in df.iterrows():
#     conversation = [{"role":"user","content":rows['prompt']}]
#     input_ids = tokenizer.apply_chat_template(
#         conversation,
#         add_generation_prompt=True,  # Add assistant generation token
#         return_tensors="pt"
#     ).to("cuda")
#     outputs = model.generate(
#         input_ids=input_ids,
#         max_new_tokens = 1000,
#         pad_token_id = tokenizer.eos_token_id
#     )
#     output.append(extract_response_llama3(tokenizer.batch_decode(outputs)[0]))
# output = pd.DataFrame(output,columns=['response'])
# output.to_csv("AD_GPT_TASK4.csv")
# %%
