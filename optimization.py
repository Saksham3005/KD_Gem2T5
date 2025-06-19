from transformers import T5ForConditionalGeneration, T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers.onnx import OnnxConfig
import os

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("outputs/t5_distilled")
tokenizer = T5Tokenizer.from_pretrained("outputs/t5_distilled")

# Convert to ONNX with quantization
os.makedirs("outputs/t5_quantized", exist_ok=True)
ort_model = ORTModelForSeq2SeqLM.from_pretrained("outputs/t5_distilled", export=True)
ort_model.quantize(
    save_dir="outputs/t5_quantized",
    quantization_config={"per_channel": True, "mode": "int8"}
)
tokenizer.save_pretrained("outputs/t5_quantized")

# Measure model size
model_size = sum(os.path.getsize(f) for f in os.listdir("outputs/t5_quantized") if not f.startswith(".")) / (1024 ** 2)  # MB
print(f"Quantized Model Size: {model_size:.2f} MB")
print("Quantized model saved to outputs/t5_quantized")