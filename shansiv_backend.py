import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(
    title="Shansiv AI",
    description="Real AI Assistant Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=API_KEY) if API_KEY else None


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "name": "Shansiv AI",
        "status": "Backend is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured"
        )

    try:
        response = client.responses.create(
            model=MODEL,
            input=request.message
        )

        return {
            "reply": response.output_text,
            "model": MODEL
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="AI request failed. Check backend logs."
        )
