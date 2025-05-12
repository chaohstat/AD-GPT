from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import BertTokenizerFast, BertForSequenceClassification,\
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
# from unsloth import FastLanguageModel
import torch
import re
import os
from utils import create_test_prompt,generate_response_bert
def get_task_class(user_input):
    MAX_LEN = 128
    # Tokenize
    encoded_input = task_tokenizer[0](
        user_input,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    encoded_input = encoded_input.to(device)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    token_type_ids = encoded_input.get('token_type_ids', None)
    if token_type_ids is not None and token_type_ids.nelement() > 0:
        token_type_ids = token_type_ids
    else:
        token_type_ids = None

    # Run inference
    with torch.no_grad():
        outputs = task_model[0](
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

# Load tokenizer and model
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = []
model_path.append("docker_app/models/base_model/question_classify_bert_v3")
model_path.append("docker_app/models/adapters/8B_finetuned_llama3.1_task1")
model_path.append("docker_app/models/base_model/task2_bert")
model_path.append("docker_app/models/adapters/8B_finetuned_llama3.1_task34_chat_v6")
task_model = []
task_tokenizer = []
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# task_tokenizer.append(BertTokenizerFast.from_pretrained(model_path[0]))
# task_model.append(BertForSequenceClassification.from_pretrained(model_path[0]))
# model_base = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

for i in range(4):
    if i in [0, 2]:
        task_model.append(BertForSequenceClassification.from_pretrained(model_path[i]).to(device))
        task_tokenizer.append(BertTokenizerFast.from_pretrained(model_path[i]))
    else:
        # Load a fresh base model each time
        model_base = AutoModelForCausalLM.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            device_map='auto',
            quantization_config=nf4_config,
            use_cache=False
        )
        peft_model = PeftModel.from_pretrained(model_base, model_path[i])  # Apply LoRA adapter
        task_model.append(peft_model)
        task_tokenizer.append(AutoTokenizer.from_pretrained(model_path[i], padding_side="right"))
from utils import extract_response_llama3
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
# Initialize FastAPI app
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend's origin for security in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Conversation history store
user_sessions = {}

# Define the request schema
class ChatRequest(BaseModel):
    user_id: str
    user_input: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    user_id = request.user_id
    user_input = request.user_input.strip()
    task_class = get_task_class(user_input)
    # Initialize user session if not present
    if user_id not in user_sessions:
        user_sessions[user_id] = []  # Start a new conversation
    # Append user input to conversation history
    user_sessions[user_id].append({"role": "user", "content": user_input})
    # Prepare the messages for the model
    messages = user_sessions[user_id]  # Current history
    print(messages)
    print(task_class)
    if task_class == 0:
        assistant_response = "invalid question, please reenter your question"
    elif task_class==1:
        input_ids = task_tokenizer[1].apply_chat_template(
            messages,
            add_generation_prompt=True,  # Add assistant generation token
            return_tensors="pt"
        ).to("cuda")
        # Generate assistant response as text
        outputs = task_model[1].generate(
            input_ids=input_ids,
            max_new_tokens=500,
            pad_token_id=task_tokenizer[1].eos_token_id
        )
        generated_ids = outputs[0].tolist()
        if generated_ids[-1] != task_tokenizer[1].eos_token_id:
            generated_ids.append(task_tokenizer[1].eos_token_id)

        # assistant_response = extract_response_llama3(tokenizer.batch_decode(outputs)[0])
        assistant_response = extract_response_llama3(task_tokenizer[1].batch_decode([generated_ids])[0])
    elif task_class==2:
        label_map = {
            0:'No',
            1:'Yes'
        }
        prompt = create_test_prompt(user_input,2)
        label = generate_response_bert(prompt,task_tokenizer[2],task_model[2])
        assistant_response = label_map[label]
    elif task_class==3:
        input_ids = task_tokenizer[3].apply_chat_template(
            messages,
            add_generation_prompt=True,  # Add assistant generation token
            return_tensors="pt"
        ).to("cuda")
        # Generate assistant response as text
        outputs = task_model[3].generate(
            input_ids=input_ids,
            max_new_tokens=5000,
            pad_token_id=task_tokenizer[3].eos_token_id
        )
        generated_ids = outputs[0].tolist()
        if generated_ids[-1] != task_tokenizer[3].eos_token_id:
            generated_ids.append(task_tokenizer[3].eos_token_id)

        # assistant_response = extract_response_llama3(tokenizer.batch_decode(outputs)[0])
        assistant_response = extract_response_llama3(task_tokenizer[3].batch_decode([generated_ids])[0])
    # Append assistant response to conversation history
    user_sessions[user_id].append({"role": "assistant", "content": assistant_response})
    user_sessions[user_id] = user_sessions[user_id][-1:]
    # Return assistant response
    return {
        "response": assistant_response,
        "conversation_history": user_sessions[user_id]
    }

    