import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk
from rouge_score import rouge_scorer
import time
import os

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("outputs/t5_distilled")
tokenizer = T5Tokenizer.from_pretrained("outputs/t5_distilled")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load test data
test_data = load_from_disk("outputs/test_data")

# Evaluation function
def evaluate_model(data, batch_size=4):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    inference_times = []
    num_examples = 0
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        articles = [data[j]["article"] for j in range(i, min(i+batch_size, len(data)))]
        targets = [data[j]["highlights"] for j in range(i, min(i+batch_size, len(data)))]
        
        # Tokenize inputs
        inputs = tokenizer(articles, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            summary_ids = model.generate(**inputs, max_length=128, num_beams=4)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Decode summaries
        summaries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
        
        # Compute ROUGE scores
        for target, summary in zip(targets, summaries):
            score = scorer.score(target, summary)
            for key in scores:
                scores[key] += score[key].fmeasure
            num_examples += 1
    
    # Average scores and times
    scores = {k: v / num_examples for k, v in scores.items()}
    avg_inference_time = sum(inference_times) / len(inference_times) * 1000  # ms per batch
    
    # Model size
    model_size = sum(os.path.getsize(f) for f in os.listdir("outputs/t5_distilled") if f.endswith(".bin")) / (1024 ** 2)  # MB
    
    print(f"ROUGE Scores: {scores}")
    print(f"Average Inference Time per Batch: {avg_inference_time:.2f} ms")
    print(f"Model Size: {model_size:.2f} MB")
    
    return scores, avg_inference_time, model_size

# Run evaluation
evaluate_model(test_data)