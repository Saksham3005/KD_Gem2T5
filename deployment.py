from fastapi import FastAPI
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import T5Tokenizer
import torch

app = FastAPI()

# Load quantized model and tokenizer
model = ORTModelForSeq2SeqLM.from_pretrained("outputs/t5_quantized")
tokenizer = T5Tokenizer.from_pretrained("outputs/t5_quantized")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Request model
class SummarizationRequest(BaseModel):
    text: str

# Summarization endpoint
@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    inputs = tokenizer(request.text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    summary_ids = model.generate(**inputs, max_length=128, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)