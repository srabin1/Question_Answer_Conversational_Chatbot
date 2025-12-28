import os
import time
import tempfile
import streamlit as st

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# ----------------------------
# Page UI
# ----------------------------
st.set_page_config(page_title="Conversational RAG with PDFs", layout="wide")
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content.")


# ----------------------------
# Optional: lightweight throttling (prevents spam)
# ----------------------------
def throttle(seconds: float = 1.5):
    now = time.time()
    last = st.session_state.get("last_call", 0.0)
    if now - last < seconds:
        st.warning(f"Please wait ~{seconds:.0f} seconds between requests.")
        st.stop()
    st.session_state["last_call"] = now


# ----------------------------
# HuggingFace embeddings (public model; HF_TOKEN optional)
# If your model needs HF token, set it via Streamlit secrets or env vars.
# ----------------------------
hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ----------------------------
# Per-user Groq API Key input
# ----------------------------
with st.sidebar:
    st.header("API Key")
    remember_key = st.checkbox("Remember key for this session", value=True)

    # If user chose to remember, pre-fill from session_state
    default_val = st.session_state.get("user_groq_key", "") if remember_key else ""

    user_api_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        value=default_val,
        help="Your key stays in your browser session only (not saved to server)."
    )

    if remember_key:
        st.session_state["user_groq_key"] = user_api_key
    else:
        st.session_state.pop("user_groq_key", None)

    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
        index=0
    )

if not user_api_key:
    st.info("Enter your Groq API key in the sidebar to start.")
    st.stop()


# ----------------------------
# Build the LLM using the user's key
# ----------------------------
llm = ChatGroq(groq_api_key=user_api_key, model_name=model_name)


# ----------------------------
# Session + chat history storage
# ----------------------------
session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


# ----------------------------
# Upload PDFs
# ----------------------------
uploaded_files = st.file_uploader(
    "Choose PDF file(s)",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload at least one PDF to build the knowledge base.")
    st.stop()


# ----------------------------
# Build vectorstore once per upload set (per user session)
# ----------------------------
upload_sig = tuple((f.name, f.size) for f in uploaded_files)

if st.session_state.get("upload_sig") != upload_sig:
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        docs = PyPDFLoader(tmp_path).load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create an in-memory Chroma vectorstore for this session
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    st.session_state.upload_sig = upload_sig

retriever = st.session_state.retriever


# ----------------------------
# RAG + History aware retriever
# ----------------------------
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# ----------------------------
# Chat UI
# ----------------------------
user_input = st.text_input("Your question:")

if user_input:
    throttle(1.5)  # optional anti-spam
    session_history = get_session_history(session_id)

    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    st.write("Assistant:", response["answer"])
    with st.expander("Chat History (debug)"):
        st.write(session_history.messages)
