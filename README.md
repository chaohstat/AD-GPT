# Bioinformatics Task Classification and Chat System

This project is a multi-turn conversational system backed by several fine-tuned LLaMA3 and BERT models. It classifies incoming bioinformatics-related questions and routes them to the appropriate model to generate accurate, context-aware responses.

---

## 🔧 Components

### 1. **FastAPI Backend (`app1.py`)**
A RESTful API that:
- Accepts user queries via a POST `/chat/` endpoint.
- Classifies the query into one of four tasks.
- Dynamically dispatches the query to a corresponding fine-tuned model (LLaMA3 or BERT).
- Maintains conversation context per user session.

### 2. **Frontend Chat Interface (`chat1.html`)**
A minimalist HTML+JS page to:
- Input user queries.
- Display responses in a chat format.
- Communicate with the FastAPI backend using `fetch`.

### 3. **Training Scripts**
#### `llama3_training.py`
- Fine-tunes the 1B LLaMA3.2 model (`unsloth/Llama-3.2-1B-bnb-4bit`) for task 1 (general bioinformatics Q&A).
- Input/output formatting using a predefined prompt template.

#### `task_classification_pipeline.py`
- Fine-tunes a BERT model to classify user questions into four task types.
- Uses HuggingFace `Trainer` with tokenization and evaluation strategy.

#### `llama3_task3.py`
- Fine-tunes the 8B LLaMA3.1 model for determining **gene–Alzheimer relationships**.
- Formats conversations from a SQLite gene database and performs multi-turn prompt generation.
- Includes BLEU evaluation for generation quality.

#### `llama3_task4.py`
- Fine-tunes the 8B LLaMA3.1 model for **mediation analysis** tasks.
- Loads preprocessed conversation data from disk and saves the final model checkpoint.

---

## 🧠 Tasks Supported

| Task ID | Description |
|---------|-------------|
|   0     | Invalid question (Filtered out) |
|   1     | General Bioinformatics QA (LLaMA3.2) |
|   2     | Binary Gene–Brain Mediation Detection (BERT) |
|   3     | Multi-turn Alzheimer Mediation Analysis (LLaMA3.1) |

---

## 🗃️ Directory Structure

```plaintext
.
├── app1.py                     # FastAPI chat server
├── chat1.html                  # Simple frontend chat UI
├── llama3_training.py          # LLaMA3 training script for Task 1
├── llama3_task3.py             # LLaMA3 training script for Task 3
├── llama3_task4.py             # LLaMA3 training script for Task 4
├── task_classification_pipeline.py  # BERT training pipeline
├── utils.py                    # Utility functions (assumed external)
├── models/                     # Saved models (referenced by FastAPI)
│   ├── base_model/
│   └── adapters/
├── data/
│   ├── task1_combined.parquet
│   ├── task2_test_v4.parquet
│   ├── gene_database_v2.db
│   └── task34_dataset_V3/
