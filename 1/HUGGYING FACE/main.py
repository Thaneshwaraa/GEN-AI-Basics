from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from model import chat

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
# Load model at server startup
@app.on_event("startup")
def load_model():
    global pipe
    model_id = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "index.html")

@app.post("/chat")
def chat(req: PromptRequest):
    response = pipe(req.prompt, max_new_tokens=256, do_sample=True)[0]["generated_text"]
    return {"response": response}