
import os
import sys
from huggingface_hub import InferenceClient

api_key = os.getenv("HF_API_KEY")
model = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Testing embedding model: {model}")
print(f"Key: {api_key[:4]}...{api_key[-4:]}")

client = InferenceClient(token=api_key)

try:
    print("Sending request...")
    # Try a simple feature extraction
    response = client.feature_extraction("Hello world", model=model)
    print(f"Success! Embedding shape: {response.shape if hasattr(response, 'shape') else len(response)}")
except Exception as e:
    print(f"\nFATAL ERROR:")
    print(str(e))
    
    # Try chat as fallback to see if key works at all
    print("\nTrying chat model as control...")
    try:
        client.chat_completion("meta-llama/Meta-Llama-3-8B-Instruct", messages=[{"role":"user","content":"hi"}])
        print("Chat works! Problem is specific to Embedding Model.")
    except:
        print("Chat also failed. Key might be invalid or rate limited.")
