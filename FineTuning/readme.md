# Text-to-SQL Fine-Tuning with QLoRA

## Project Overview

This project implements a Text-to-SQL conversion model using Quantized Low-Rank Adaptation (QLoRA) fine-tuning. The goal is to generate accurate SQL queries from natural language questions by leveraging the Salesforce CodeGen2-7B model and fine-tuning it on the Spider and WikiSQL datasets.

## Key Features

- Utilizes Salesforce CodeGen2-7B as the base model
- Implements QLoRA for efficient fine-tuning
- Combines Spider and WikiSQL datasets for training
- Supports generating SQL queries from natural language questions

## Dependencies

- transformers
- datasets
- accelerate
- peft
- bitsandbytes
- torch

## Installation

```bash
pip install -q -U transformers datasets accelerate peft bitsandbytes torch
```

## Model Training Process

The fine-tuning process involves:
1. Loading and preparing datasets (Spider and WikiSQL)
2. Creating a QLoRA configuration
3. Fine-tuning the model on text-to-SQL task
4. Saving the fine-tuned model

### Key Hyperparameters

- Base Model: Salesforce CodeGen2-7B
- LoRA Rank (r): 8
- LoRA Alpha: 32
- LoRA Dropout: 0.05
- Epochs: 3
- Batch Size: 4

## Usage Example

```python
# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./qlora-text-to-sql-final")
model = AutoModelForCausalLM.from_pretrained("./qlora-text-to-sql-final")

# Generate SQL query
def generate_sql(question, max_new_tokens=128):
    prompt = f"Question: {question}\nGenerate the SQL query:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
question = "List the name and age of all students older than 20."
sql_query = generate_sql(question)
print(sql_query)
```

## Acknowledgments

- Salesforce for the CodeGen2-7B model
- Hugging Face Transformers library
- Spider and WikiSQL dataset creators
