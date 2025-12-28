import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

# ----------------------------
# LangSmith Tracking (optional)
# ----------------------------
# NOTE: You had a bug: you were setting LANGCHAIN_API_KEY from OPENAI_API_KEY.
langsmith_key = (os.getenv("LANGCHAIN_API_KEY") or st.secrets.get("LANGCHAIN_API_KEY") or "").strip()
if langsmith_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With OpenAI"

# ----------------------------
# Prompt Template
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, model_name, temperature, max_tokens):
    llm_client = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_parser = StrOutputParser()
    chain = prompt | llm_client | output_parser
    return chain.invoke({"question": question})

# Title
st.title("Enhanced Q&A Chatbot With OpenAI")

# Sidebar
st.sidebar.title("Settings")

# Keep key stable across Streamlit reruns
if "user_openai_key" not in st.session_state:
    st.session_state.user_openai_key = ""

api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    key="user_openai_key",
    help="Stored only in your current browser session."
)
api_key = (api_key or "").strip()

model_name = st.sidebar.selectbox(
    "Select OpenAI model",
    ["gpt-4o", "gpt-4.1-mini", "gpt-4o-mini", "gpt-4-turbo"],
    index=0
)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input and api_key:
    try:
        response = generate_response(user_input, api_key, model_name, temperature, max_tokens)
        st.write(response)
    except Exception:
        st.error("Request failed. Please verify your API key and model selection.")
elif user_input:
    st.warning("Please enter the OpenAI API Key in the sidebar")
else:
    st.write("Please provide the user input")
