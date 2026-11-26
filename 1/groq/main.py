import os, logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, RateLimitError, AuthenticationError, APIError

load_dotenv()
logger = logging.getLogger("uvicorn.error")

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env")

MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")  # llama-3.3-70b-versatile was removed by Groq
client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1",
                     api_key=API_KEY, timeout=30)

app = FastAPI(title="Groq Chat API")

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    temperature: float = Field(0.7, ge=0, le=2)

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: PromptRequest):
    try:
        r = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
                {"role": "user", "content": req.prompt},
            ],
            temperature=req.temperature,
            max_tokens=500,
        )
    except RateLimitError:
        raise HTTPException(429, "Too many requests, try again later")
    except AuthenticationError:
        logger.error("Invalid Groq API key")
        raise HTTPException(500, "Server configuration error")
    except APIError as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(502, "LLM service error")
    return ChatResponse(response=r.choices[0].message.content)