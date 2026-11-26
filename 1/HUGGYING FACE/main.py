from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

from model import chat  # model is loaded once, when model.py is imported

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat")
def chat_endpoint(req: PromptRequest):
    return {"response": chat(req.prompt)}
