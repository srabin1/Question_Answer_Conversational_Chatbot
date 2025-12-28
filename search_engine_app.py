import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
import os


# ----------------------------
# Tools (unchanged)
# ----------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")


# ----------------------------
# UI
# ----------------------------
st.title("🔎 LangChain - Chat with search")
st.write(
    "This project uses `StreamlitCallbackHandler` to display agent thoughts/actions in Streamlit."
)

# Sidebar for settings
st.sidebar.title("Settings")

# Keep key stable across reruns (prevents blank/overwrite issues)
if "user_groq_key" not in st.session_state:
    st.session_state.user_groq_key = ""

api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password", key="user_groq_key")
api_key = (api_key or "").strip()

model_name = st.sidebar.selectbox(
    "Groq model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
    index=0
)

# Optional local fallback for your machine (keeps your local workflow intact)
if not api_key:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()

if not api_key:
    st.info("Please enter your Groq API key in the sidebar to start.")
    st.stop()


# ----------------------------
# Chat history
# ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ----------------------------
# Chat input
# ----------------------------
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # LLM uses the user's key
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=False)

    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": (
                "You are a tool-using agent. Always use the format:\n"
                "Thought:\nAction:\nAction Input:\n... and end with:\nFinal:\n"
                "Never output XML tags or JSON unless it is Action Input."
            )
        },
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            result = search_agent.invoke({"input": prompt}, callbacks=[st_cb])
            response = result.get("output", "")
        except Exception as e:
            st.error("The agent failed to respond. Please verify your key/model and try again.")
            st.stop()

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
