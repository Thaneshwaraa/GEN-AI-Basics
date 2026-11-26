from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Connect to local Ollama (OpenAI-compatible)
model = OpenAIChatModel(
    "llama3.2",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
)

# Structured output: the LLM must return this shape
class Answer(BaseModel):
    answer: str
    confidence: str   # low / medium / high

agent = Agent(model, output_type=Answer,
              system_prompt="Answer briefly and give your confidence.")

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: PromptRequest):
    result = await agent.run(req.prompt)
    return result.output      # already a validated Answer object