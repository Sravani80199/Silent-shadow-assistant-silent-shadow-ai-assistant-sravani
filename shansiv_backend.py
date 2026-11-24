
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(
    title="Silent Shadow Assistant",
    description="Real OpenAI AI Backend",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=API_KEY) if API_KEY else None


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "name": "Silent Shadow Assistant",
        "status": "Backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ai_configured": client is not None
    }


@app.post("/chat")
def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a message."
        )

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured."
        )

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=(
                "You are Silent Shadow Assistant, "
                "a helpful AI assistant for students, "
                "developers and researchers. "
                "Give clear, accurate and practical answers."
            ),
            input=request.message
        )

        return {
            "reply": response.output_text,
            "model": MODEL,
            "status": "success"
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="AI request failed. Check backend logs and API access."
        )
        
