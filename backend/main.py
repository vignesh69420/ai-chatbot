from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from openai import AsyncOpenAI
import os

app = FastAPI()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/api/chat")
async def chat(req: ChatRequest):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": m.role, "content": m.content} for m in req.messages],
    )
    return {"message": response.choices[0].message.content}
