import os
from datasets import load_dataset
from transformers import T5Tokenizer
import torch

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Load T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load CNN/Daily Mail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Preprocess function
def preprocess(example):
    article = example["article"]
    summary = example["highlights"]
    # Tokenize inputs and targets
    inputs = tokenizer(article, max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(summary, max_length=128, truncation=True, padding="max_length")
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": targets.input_ids
    }

# Apply preprocessing
train_data = dataset["train"].select(range(1000))  # Subset for demo
test_data = dataset["test"].select(range(200))
train_data = train_data.map(preprocess, batched=False)
test_data = test_data.map(preprocess, batched=False)

# Save preprocessed data
train_data.save_to_disk("outputs/train_data")
test_data.save_to_disk("outputs/test_data")

print("Data preparation complete. Preprocessed data saved to outputs/train_data and outputs/test_data.")