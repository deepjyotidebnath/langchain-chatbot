"""
Simple LangChain + Streamlit Chatbot
-------------------------------------
A minimal conversational chatbot with memory, powered by LangChain and Groq.

Setup:
    pip install -r requirements.txt
    Copy .env.example to .env and add your Groq API key (get one free at https://console.groq.com/keys)
    Run: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

st.set_page_config(page_title="LangChain Chatbot", page_icon="💬")
st.title("💬 Simple LangChain Chatbot")
st.caption("Built with Streamlit + LangChain + Groq")

# --- Sidebar: settings ---
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        index=0,
    )
    if st.button("Clear conversation"):
        st.session_state.clear()
        st.rerun()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Add it to a .env file in this folder (see .env.example).")
    st.stop()

# --- Initialize LangChain LLM + message history (once per session) ---
if "chain" not in st.session_state:
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.7)
    history_store = {"session": InMemoryChatMessageHistory()}
    st.session_state.chain = RunnableWithMessageHistory(
        llm,
        lambda session_id: history_store["session"],
    )
    st.session_state.messages = []

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke(
                [{"role": "user", "content": user_input}],
                config={"configurable": {"session_id": "session"}},
            )
            response = result.content
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})