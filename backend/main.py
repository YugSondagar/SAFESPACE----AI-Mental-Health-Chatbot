from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
# Import parse_response from your ai_agents.py
from ai_agents import parse_response

app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/ask")
async def ask(query: Query):
    try:
        # This calls the AgentExecutor and returns (tool_name, final_answer)
        tool_called_name, final_response = parse_response(query.message)
        
        return {
            "response": final_response,
            "tool_called": tool_called_name
        }
    except Exception as e:
        # Catch errors so the frontend doesn't get a "JSONDecodeError"
        print(f"Error occurred: {e}")
        return {
            "response": "I'm sorry, I encountered an error processing that request.",
            "tool_called": "Error"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)