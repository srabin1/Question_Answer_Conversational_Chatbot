import os
import time
import streamlit as st

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


st.title("RAG Document Q&A With Groq")

# ----------------------------
# Per-user API keys (UI input)
# ----------------------------
with st.sidebar:
    st.header("API Keys (your own)")
    st.caption("Keys are stored only in your current browser session.")

    if "user_openai_key" not in st.session_state:
        st.session_state.user_openai_key = ""
    if "user_groq_key" not in st.session_state:
        st.session_state.user_groq_key = ""

    openai_key = st.text_input("OpenAI API key (for embeddings)", type="password", key="user_openai_key")
    groq_key = st.text_input("Groq API key (for chat)", type="password", key="user_groq_key")

    model_name = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
        index=0
    )

openai_key = (openai_key or "").strip()
groq_key = (groq_key or "").strip()

# Optional: allow local env vars as fallback (keeps local workflow intact)
if not openai_key:
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
if not groq_key:
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()

if not groq_key:
    st.info("Enter your **Groq API key** in the sidebar to start.")
    st.stop()

if not openai_key:
    st.info("Enter your **OpenAI API key** in the sidebar (needed for OpenAIEmbeddings).")
    st.stop()

# Set env vars for the libraries that expect them
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["GROQ_API_KEY"] = groq_key

# --- LLM ---
llm = ChatGroq(
    groq_api_key=groq_key,
    model_name=model_name
)

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based only on the provided context.
Provide the most accurate response.

<context>
{context}
</context>

Question: {input}
"""
)

st.write('1) Click **Document Embedding** once to build the vector DB.')
st.write('2) Then ask questions.')


# ---- Build vector DB once (cached) ----
# IMPORTANT: include the OPENAI key as an argument so cache is per-user-key
@st.cache_resource(show_spinner=True)
def build_vector_db(pdf_folder: str, openai_api_key_for_cache: str):
    # Ensure this function uses the correct key (important on Streamlit Cloud)
    os.environ["OPENAI_API_KEY"] = openai_api_key_for_cache

    embeddings = OpenAIEmbeddings()
    loader = PyPDFDirectoryLoader(pdf_folder)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectors = FAISS.from_documents(chunks, embeddings)
    return vectors


if "vectors" not in st.session_state:
    st.session_state.vectors = None

if st.button("Document Embedding"):
    try:
        st.session_state.vectors = build_vector_db("research_papers", openai_key)
        st.success("Vector Database is ready ✅")
    except Exception as e:
        st.error(f"Failed to build vector DB: {e}")
        st.stop()


# Use a form so Enter submits in a controlled way
with st.form("qa_form"):
    user_prompt = st.text_input(
        'Enter your query from the research paper "Attention Is All You Need" and "A Comprehensive Overview of Large Language Models"'
    )
    ask = st.form_submit_button("Ask")

if ask:
    if st.session_state.vectors is None:
        st.warning("Please click **Document Embedding** first to create the vector database.")
        st.stop()

    if not user_prompt.strip():
        st.warning("Please type a question.")
        st.stop()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.caption(f"Response time: {time.process_time() - start:.3f} seconds")

    st.write(response.get("answer", ""))

    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response.get("context", []), start=1):
            st.write(f"Chunk {i}:")
            st.write(doc.page_content)
            st.write("---")
