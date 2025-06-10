# app/hf_client.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Save memory
    device_map="cpu"           # Auto-distribute across CPU/GPU
)

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1
)

def query_hf_llm(prompt: str):
    """Generate response from HF model."""
    logger.info("[HF] Generating response...")
    try:
        full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        output = pipe(full_prompt)
        generated_text = output[0]["generated_text"]
        
        # Extract only assistant's part
        if "<|assistant|>" in generated_text:
            answer = generated_text.split("<|assistant|>")[-1].strip()
        else:
            answer = generated_text.strip()
            
        return answer
    except Exception as e:
        logger.error(f"[HF] Error generating response: {e}")
        return f"[ERROR] Model failed: {e}"