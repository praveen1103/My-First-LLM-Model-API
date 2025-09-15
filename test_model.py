from transformers import pipeline

# Load the model
generator = pipeline("text-generation", model="distilgpt2")

# Test the model
prompt = "Hello, Explain Newton laws"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]["generated_text"])