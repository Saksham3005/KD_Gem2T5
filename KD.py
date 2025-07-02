import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk
import google.generativeai as genai
from keys import key

# genai.configure(api_key="your_real_api_key_here")


# Configure Google AI API (replace with your API key)
genai.configure(api_key=key)  # Obtain from https://ai.google.dev
teacher_model = genai.GenerativeModel("gemini-1.5-flash")

# Load T5 model and tokenizer
student_model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(device)

# Load preprocessed data
train_data = load_from_disk("outputs/train_data")

# Get teacher summaries
def get_teacher_summary(text):
    try:
        response = teacher_model.generate_content(f"Summarize: {text}")
        return response.text
    except Exception as e:
        print(f"API error: {e}")
        return ""

# Training loop
def train_student(data, epochs=10, alpha=0.5, batch_size=3):
    optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)
    student_model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            articles = [data[j]["article"] for j in range(i, min(i+batch_size, len(data)))]
            target_summaries = [data[j]["highlights"] for j in range(i, min(i+batch_size, len(data)))]
            
            # Get teacher summaries
            teacher_summaries = [get_teacher_summary(article) for article in articles]
            teacher_summaries = [s if s else target_summaries[k] for k, s in enumerate(teacher_summaries)]
            
            # Tokenize
            inputs = tokenizer(articles, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
            teacher_ids = tokenizer(teacher_summaries, return_tensors="pt", max_length=128, truncation=True, padding=True).input_ids.to(device)
            target_ids = tokenizer(target_summaries, return_tensors="pt", max_length=128, truncation=True, padding=True).input_ids.to(device)
            
            # Forward pass
            outputs = student_model(**inputs, labels=teacher_ids)
            dist_loss = outputs.loss
            task_loss = student_model(**inputs, labels=target_ids).loss
            loss = alpha * dist_loss + (1 - alpha) * task_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i // batch_size) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i//batch_size}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Average Loss: {total_loss / (len(data) // batch_size):.4f}")
    
    # Save model
    student_model.save_pretrained("outputs/t5_distilled")
    tokenizer.save_pretrained("outputs/t5_distilled")
    print("Model saved to outputs/t5_distilled")

# Run training
train_student(train_data)