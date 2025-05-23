import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast,BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import Dataset
import evaluate
from transformers import TrainingArguments, Trainer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7"
metric = evaluate.load("accuracy")


excel_df = pd.read_excel('supp.xlsx', sheet_name=8, usecols="A:B", header=1)
gene_name = excel_df['Gene symbol'].tolist()
gene_name = [' '.join(name.split()) for name in gene_name]
gene_name = ['SGMS1' if name == 'SMS1' else name for name in gene_name]
gene_name = ['MTOR' if name == 'mTOR' else name for name in gene_name]

# Load your dataset
# df = pd.read_parquet('question_v4.parquet')  # Replace with your data loading method
# df = pd.read_parquet('task2_v4.parquet')
df = pd.read_parquet('task2_test_v4.parquet')
df['Response'] = df['Response'].map({'Yes': 1, 'No': 0})
model_name = 'google-bert/bert-base-uncased'
dataset = Dataset.from_pandas(df)
splitted_datasets = dataset.train_test_split(test_size=0.3)

tokenizer = BertTokenizerFast.from_pretrained(model_name)
tokenizer.add_tokens(gene_name)
MAX_LEN = 128  # Adjust based on your data
BATCH_SIZE = 16

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4  # Number of classes
)
model.resize_token_embeddings(len(tokenizer))
def preprocess_function_batch(examples):
    # truncation=True: truncate to the maximum acceptable input length for
    # the model. 
    # return tokenizer(examples["Input"], truncation=True)
    return tokenizer(examples["question"], truncation=True)
splitted_datasets_encoded = splitted_datasets.map(preprocess_function_batch, batched=True)
splitted_datasets_encoded = splitted_datasets_encoded.rename_column("task_type", "label")
# splitted_datasets_encoded = splitted_datasets_encoded.rename_column("Response", "label")

# model_output_dir = '/nas1/llm/task2/bert'
model_output_dir = '/nas1/llm/question_class/bert'
args = TrainingArguments(
    # output_dir: directory where the model checkpoints will be saved.
    output_dir=model_output_dir,
    # evaluation_strategy (default "no"):
    # Possible values are:
    # "no": No evaluation is done during training.
    # "steps": Evaluation is done (and logged) every eval_steps.
    # "epoch": Evaluation is done at the end of each epoch.
    evaluation_strategy="steps",
    # eval_steps: Number of update steps between two evaluations if
    # evaluation_strategy="steps". Will default to the same value as
    # logging_steps if not set.
    eval_steps=50,
    # logging_strategy (default: "steps"): The logging strategy to adopt during
    # training (used to log training loss for example). Possible values are:
    # "no": No logging is done during training.
    # "epoch": Logging is done at the end of each epoch.
    # "steps": Logging is done every logging_steps.
    logging_strategy="steps",
    # logging_steps (default 500): Number of update steps between two logs if
    # logging_strategy="steps".
    logging_steps=50,
    # save_strategy (default "steps"):
    # The checkpoint save strategy to adopt during training. Possible values are:
    # "no": No save is done during training.
    # "epoch": Save is done at the end of each epoch.
    # "steps": Save is done every save_steps (default 500).
    save_strategy="steps",
    # save_steps (default: 500): Number of updates steps before two checkpoint
    # saves if save_strategy="steps".
    save_steps=50,
    # learning_rate (default 5e-5): The initial learning rate for AdamW optimizer.
    # Adam algorithm with weight decay fix as introduced in the paper
    # Decoupled Weight Decay Regularization.
    learning_rate=2e-5,
    # per_device_train_batch_size: The batch size per GPU/TPU core/CPU for training.
    per_device_train_batch_size=16,
    # per_device_eval_batch_size: The batch size per GPU/TPU core/CPU for evaluation.
    per_device_eval_batch_size=16,
    # num_train_epochs (default 3.0): Total number of training epochs to perform
    # (if not an integer, will perform the decimal part percents of the last epoch
    # before stopping training).
    num_train_epochs=5,
    # load_best_model_at_end (default False): Whether or not to load the best model
    # found during training at the end of training.
    load_best_model_at_end=True,
    # metric_for_best_model:
    # Use in conjunction with load_best_model_at_end to specify the metric to use
    # to compare two different models. Must be the name of a metric returned by
    # the evaluation with or without the prefix "eval_".
    metric_for_best_model="accuracy",
    # report_to:
    # The list of integrations to report the results and logs to. Supported
    # platforms are "azure_ml", "comet_ml", "mlflow", "tensorboard" and "wandb".
    # Use "all" to report to all integrations installed, "none" for no integrations.
    # report_to="tensorboard"
)
def model_init():
    model = BertForSequenceClassification.from_pretrained(model_name,
                                                              num_labels=4)
    model.resize_token_embeddings(len(tokenizer))
    return model

# Function that will be called at the end of each evaluation phase on the whole
# arrays of predictions/labels to produce metrics.
def compute_metrics(eval_pred):
    # Predictions and labels are grouped in a namedtuple called EvalPrediction
    predictions, labels = eval_pred
    # Get the index with the highest prediction score (i.e. the predicted labels)
    predictions = np.argmax(predictions, axis=1)
    # Compare the predicted labels with the reference labels
    results =  metric.compute(predictions=predictions, references=labels)
    # results: a dictionary with string keys (the name of the metric) and float
    # values (i.e. the metric values)
    return results

# Since PyTorch does not provide a training loop, the 🤗 Transformers library
# provides a Trainer API that is optimized for 🤗 Transformers models, with a
# wide range of training options and with built-in features like logging,
# gradient accumulation, and mixed precision.
trainer = Trainer(
    # Function that returns the model to train. It's useful to use a function
    # instead of directly the model to make sure that we are always training
    # an untrained model from scratch.
    model_init=model_init,
    # The training arguments.
    args=args,
    # The training dataset.
    train_dataset=splitted_datasets_encoded["train"],
    # The evaluation dataset. We use a small subset of the validation set
    # composed of 150 samples to speed up computations...
    eval_dataset=splitted_datasets_encoded["test"].shuffle(42).select(range(150)),
    # Even though the training set and evaluation set are already tokenized, the
    # tokenizer is needed to pad the "input_ids" and "attention_mask" tensors
    # to the length managed by the model. It does so one batch at a time, to
    # use less memory as possible.
    tokenizer=tokenizer,
    # Function that will be called at the end of each evaluation phase on the whole
    # arrays of predictions/labels to produce metrics.
    compute_metrics=compute_metrics
)

# ... train the model!
trainer.train()
# trainer.save_model("/nas1/task2_bert_v4")
trainer.save_model("/nas1/llm/question_classify_bert_v3")
tokenizer.save_pretrained('/nas1/llm/question_classify_bert_v3')