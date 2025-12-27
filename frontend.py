import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="AI Mental Health Therapist", layout="wide")
st.title("🧠 SafeSpace – AI Mental Health Therapist")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("What's on your mind today?")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    try:
        response = requests.post(BACKEND_URL, json={"message": user_input})
        
        if response.status_code == 200:
            data = response.json()
            final_content = data["response"]
            tool_name = data["tool_called"]
            
            # Append AI response with the tool info
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"{final_content}\n\n*System Note: Used tool [{tool_name}]*"
            })
        else:
            st.error("The backend is running but returned an error.")
            
    except Exception as e:
        st.error(f"Could not connect to backend. Is main.py running? Error: {e}")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])