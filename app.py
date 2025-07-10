import os
import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv
from retriever import create_retriever,get_context
from langchain_groq import ChatGroq
from prompts import prompt_template

load_dotenv('.env')

api_key = os.getenv('GROQ_API_KEY')



if not api_key:
    st.error("API key not found.")
    st.stop()

# ---- UI HEADER ----
st.title("📄 Policy Chatbot")

# ---- SESSION STATE SETUP ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- RETRIEVER INITIALIZATION ----

try:
    retriever = create_retriever()
except Exception as e:
    st.error(f"Error initializing retriever: {str(e)}")
    st.stop()

# ---- RAG PIPELINE ----
def rag_pipeline(question):
    llm = ChatGroq(
		model="llama3-70b-8192",
		temperature=0.2,
		api_key=api_key
	)
    
    context = get_context(question)
    
    
    prompt = prompt_template.replace("{context}", context).replace("{question}", question)
    
    response = llm.invoke(prompt)
    return response.content

# ---- CHAT DISPLAY ----
# New Chat button
if st.button("🔄 New Chat"):
		st.session_state.chat_history = []
		st.rerun()
  
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
        st.caption(f"{chat['user_time']}")

    with st.chat_message("assistant"):
        st.markdown(chat["bot"])
        st.caption(f"{chat['bot_time']}")

# Input and response generation
if user_input := st.chat_input("Ask anything..."):
    user_time = datetime.now().strftime("%I:%M %p")
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(f"{user_time}")

    with st.spinner("Thinking..."):
        bot_response = rag_pipeline(user_input)
        bot_time = datetime.now().strftime("%I:%M %p")

    with st.chat_message("assistant"):
        st.markdown(bot_response)
        st.caption(f"{bot_time}")

    # Save both timestamps in chat history
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": bot_response,
        "user_time": user_time,
        "bot_time": bot_time
    })