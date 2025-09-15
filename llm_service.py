from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

# Load the model (loaded once at startup)
generator = pipeline("text-generation", model="distilgpt2")

# Define input model
class TextInput(BaseModel):
    prompt: str
    max_length: int = 50

# Endpoint for text generation
@app.post("/generate")
async def generate_text(input: TextInput):
    result = generator(input.prompt, max_length=input.max_length, num_return_sequences=1)
    return {"generated_text": result[0]["generated_text"]}