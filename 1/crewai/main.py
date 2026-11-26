from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM

llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

researcher = Agent(
    role="Researcher",
    goal="Find key facts about the topic",
    backstory="An expert who finds accurate information.",
    llm=llm,
)
writer = Agent(
    role="Writer",
    goal="Write a short, simple explanation",
    backstory="A writer who explains things simply.",
    llm=llm,
)

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat(req: PromptRequest):
    research = Task(description=f"Research: {req.prompt}",
                    expected_output="5 key bullet points", agent=researcher)
    write = Task(description="Write a short explanation from the research",
                 expected_output="One short paragraph", agent=writer)

    crew = Crew(agents=[researcher, writer], tasks=[research, write])
    result = crew.kickoff()
    return {"response": str(result)}