# KD_Gem2T5 #  

Black-Box Knowledge Distillation: Gemini 1.5 Flash to T5 for Text Summarization

## Project Overview
This project implements **black-box knowledge distillation (KD)** to transfer the text summarization capabilities of **Gemini 1.5 Flash** (accessed via Google AI API) into a lightweight **T5-small** model. The task is abstractive text summarization on the **CNN/Daily Mail dataset**, chosen for its well-defined structure and compatibility with both models. The distilled T5-small model is optimized for efficiency (via quantization) and deployed as a web-based demo using FastAPI, making it suitable for resource-constrained environments.

The project addresses the challenge of differing vocabulary sizes between Gemini 1.5 Flash and T5 by using **text-based distillation**, where T5 is trained to mimic teacher-generated summaries, bypassing logits mismatches. This approach leverages T5’s SentencePiece tokenizer for consistent input/output processing. The project is designed to enhance my portfolio by showcasing skills in NLP, model compression, API integration, and deployment, while deepening my understanding of knowledge distillation.

## Objectives
- Distill Gemini 1.5 Flash’s summarization knowledge into T5-small.
- Achieve high ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) on CNN/Daily Mail.
- Optimize the student model for size and inference speed.
- Deploy a web-based summarization demo.
- Open-source the codebase for community use.

## Methodology
### Task
- **Abstractive Text Summarization**: Generate concise summaries of news articles from the CNN/Daily Mail dataset.
- **Teacher Model**: Gemini 1.5 Flash (via Google AI API).
- **Student Model**: T5-small (Hugging Face Transformers).
- **Dataset**: CNN/Daily Mail (available via Hugging Face Datasets).

### Black-Box KD Approach
- **Text-Based Distillation**: Query Gemini 1.5 Flash to generate summaries for training articles. Train T5-small to predict these summaries using sequence-to-sequence loss, combined with task-specific loss (ground-truth summaries).
- **Vocabulary Mismatch Solution**: Use T5’s SentencePiece tokenizer for all inputs and outputs, ensuring consistency and avoiding logits-based distillation, which could be problematic due to differing vocabularies. If logits were accessible, a linear mapping layer could align vocabularies (not implemented here due to API constraints).

### Project Phases
1. **Data Preparation**: Load and preprocess CNN/Daily Mail dataset, tokenize with T5’s tokenizer.
2. **Knowledge Distillation**: Train T5-small to mimic Gemini 1.5 Flash summaries.
3. **Evaluation**: Measure ROUGE scores, inference time, and model size.
4. **Optimization**: Apply 8-bit quantization to reduce model size.
5. **Deployment**: Create a FastAPI web app for summarization.

## Repository Structure
The repository contains the following scripts, each corresponding to a project phase:

- **`data_preparation.py`**: Loads CNN/Daily Mail dataset, tokenizes it with T5’s tokenizer, and saves preprocessed data to `./outputs`.
- **`knowledge_distillation.py`**: Queries Gemini 1.5 Flash for summaries and trains T5-small using text-based distillation.
- **`evaluation.py`**: Evaluates the distilled model on ROUGE metrics and measures efficiency (inference time, model size).
- **`optimization.py`**: Applies 8-bit quantization to T5-small using ONNX.
- **`deployment.py`**: Deploys the quantized model as a FastAPI web app.

## Setup Instructions
### Prerequisites
- Python 3.8+
- Google AI API key (obtain from [https://ai.google.dev](https://ai.google.dev))
- Dependencies:
  ```bash
  pip install torch transformers datasets google-generativeai rouge_score fastapi uvicorn optimum onnx
  ```

### Environment Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies (see above).
3. Set up your Google AI API key:
   - Replace `YOUR_API_KEY` in `knowledge_distillation.py` with your API key.

### Running the Scripts
1. **Data Preparation**:
   ```bash
   python data_preparation.py
   ```
   - Outputs: Preprocessed datasets in `./outputs/train_data` and `./outputs/test_data`.
2. **Knowledge Distillation**:
   ```bash
   python knowledge_distillation.py
   ```
   - Outputs: Trained T5-small model in `./outputs/t5_distilled`.
   - Note: Ensure API key is set and internet connection is active.
3. **Evaluation**:
   ```bash
   python evaluation.py
   ```
   - Outputs: ROUGE scores, inference time, and model size (printed to console).
4. **Optimization**:
   ```bash
   python optimization.py
   ```
   - Outputs: Quantized model in `./outputs/t5_quantized`.
5. **Deployment**:
   ```bash
   python deployment.py
   ```
   - Access the API at `http://localhost:8000`. Test with a POST request to `/summarize`:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"text": "Your article text here"}' http://localhost:8000/summarize
     ```

## Usage
- **Training**: Run `data_preparation.py` followed by `knowledge_distillation.py` to train the model. A subset of 1000 training and 200 test examples is used for demo purposes; adjust in `data_preparation.py` for larger datasets.
- **Evaluation**: Run `evaluation.py` to assess model performance. Expect ROUGE scores competitive with T5-small baselines and reduced inference time post-optimization.
- **Deployment**: Use `deployment.py` to serve the model. The API accepts a JSON payload with a `text` field and returns a summary.

## Handling Vocabulary Mismatches
Gemini 1.5 Flash and T5-small use different tokenizers and vocabularies, which could complicate logits-based distillation. This project uses **text-based distillation** to avoid this issue:
- **Input Processing**: Articles are tokenized with T5’s SentencePiece tokenizer before querying Gemini 1.5 Flash.
- **Output Processing**: Gemini-generated summaries are re-tokenized with T5’s tokenizer for training, ensuring vocabulary consistency.
- **Alternative (Not Used)**: If logits were accessible, a linear layer could map Gemini’s logits to T5’s vocabulary. Text-based distillation was chosen for simplicity and API compatibility.

## Results
- **Expected ROUGE Scores**: Comparable to T5-small baselines (e.g., ROUGE-1 ~0.40, ROUGE-2 ~0.20, ROUGE-L ~0.35 on CNN/Daily Mail).
- **Efficiency**: Quantized T5-small reduces model size by ~50% (from ~240 MB to ~120 MB) and improves inference speed on CPU.
- **Demo**: The FastAPI app provides a user-friendly interface for summarization, suitable for portfolio showcases.

## Future Improvements
- Experiment with larger T5 variants (e.g., T5-base) or other datasets (e.g., XSum).
- Implement logits-based distillation if Google AI API exposes logits.
- Enhance the web demo with a front-end interface (e.g., HTML/React).
- Explore advanced optimization techniques (e.g., pruning, dynamic quantization).
