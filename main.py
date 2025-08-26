from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os

# Import the RAG agent
try:
    from rag_agent import app as rag_app
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"RAG agent import error: {e}")
    RAG_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG Chatbot",
    description="A chatbot powered by Agentic RAG to answer questions about Edhi Hospital and other topics",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the main chatbot interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """Handle chat messages and return RAG-powered responses"""
    try:
        if not chat_request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if RAG_AVAILABLE:
            try:
                # Use the existing RAG workflow
                result = rag_app.invoke({"messages": [chat_request.message]})
                
                # Extract the response from the RAG workflow
                messages = result.get("messages", [])
                if messages:
                    # Extract only the content from the response, not the metadata
                    last_message = str(messages[-1])
                    if "content='" in last_message:
                        # Extract content between content=' and the next quote
                        start = last_message.find("content='") + 9
                        end = last_message.find("'", start)
                        if end != -1:
                            response = last_message[start:end]
                        else:
                            response = last_message
                    else:
                        response = last_message
                else:
                    response = "I'm sorry, I couldn't find specific information about that. Please try asking about Edhi Hospital doctors, services, or facilities."
                
            except Exception as rag_error:
                response = f"I'm having trouble accessing the hospital database right now. Please try again in a moment. (Error: {str(rag_error)})"
        else:
            response = "I'm sorry, the RAG system is not available at the moment. Please check back soon!"
        
        return ChatResponse(
            response=response,
            status="success"
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Agentic RAG Chatbot",
        "rag_available": RAG_AVAILABLE
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8007,
        reload=True,
        log_level="info"
    )
