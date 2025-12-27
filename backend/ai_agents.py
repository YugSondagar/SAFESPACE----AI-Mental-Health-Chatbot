from langchain.tools import tool
from tools import query_medgemma, call_emergency
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from config import GROQ_API_KEY

# ----------------- Tools -----------------

@tool
def ask_mental_health_specialist(query: str) -> str:
    """
    Generate a therapeutic response using the MedGemma model.
    Use this for all general user queries, mental health questions, emotional concerns,
    or to offer empathetic, evidence-based guidance in a conversational tone.
    """
    return query_medgemma(query)

@tool
def emergency_call_tool(reason: str) -> str:
    """
    Place an emergency call via Twilio. 
    Use this ONLY if the user expresses suicidal ideation or intent to self-harm.
    Input MUST be a brief description of the emergency (e.g., 'User expressing self-harm intent').
    """
    call_emergency()
    return "Emergency services have been contacted via Twilio. Help is on the way."

@tool
def find_nearby_therapists_by_location(location: str) -> str:
    """
    Finds and returns a list of licensed therapists near the specified location.
    Args:
        location (str): The name of the city or area.
    """
    return f"Therapists near {location}: [List of local clinics and contact info]"

# ----------------- LLM & Agent -----------------

tools = [
    ask_mental_health_specialist,
    emergency_call_tool,
    find_nearby_therapists_by_location,
]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,  # Lower temperature for stricter adherence to format
    api_key=GROQ_API_KEY,
)

# Strict ReAct Template to prevent StopIteration and formatting errors
template = """
You are a compassionate AI Mental Health Assistant. Your goal is to support the user.
You have access to the following tools:

{tools}

To use a tool, you MUST use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a final response for the user, or if you do not need to use a tool, you MUST use this format:

Thought: Do I need to use a tool? No
Final Answer: [your warm, empathetic response here]

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# ✅ Initialize the agent with the strict prompt
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# ✅ AgentExecutor handles the loop and catches parsing slips
graph = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=5
)

# ----------------- Response Parser -----------------

def parse_response(user_input):
    """
    Calls the AgentExecutor and extracts the final output and tool used.
    """
    try:
        # Use invoke to get the complete dictionary back
        response = graph.invoke({"input": user_input})
        
        # 'output' contains the string from 'Final Answer'
        final_response = response.get("output", "I'm sorry, I'm having trouble formulating a response.")
        
        # Extract tool name from intermediate steps if a tool was used
        steps = response.get("intermediate_steps", [])
        if steps:
            tool_called_name = steps[-1][0].tool
        else:
            tool_called_name = "None (Direct Response)"

        return tool_called_name, final_response

    except Exception as e:
        print(f"Error in parse_response: {e}")
        return "Error", "I encountered a formatting issue. I am here to listen; please tell me more."