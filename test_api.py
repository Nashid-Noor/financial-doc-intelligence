
import os
import time
from huggingface_hub import InferenceClient

api_key = os.getenv("HF_API_KEY")
print(f"Testing with key: {api_key[:4]}...{api_key[-4:]}")

client = InferenceClient(token=api_key)
model = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Pinging model: {model}...")
start = time.time()
try:
    response = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": "Hello, are you working?"}],
        max_tokens=50
    )
    duration = time.time() - start
    print(f"Success! Time: {duration:.2f}s")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    duration = time.time() - start
    print(f"Failed after {duration:.2f}s")
    print(f"Error: {e}")
