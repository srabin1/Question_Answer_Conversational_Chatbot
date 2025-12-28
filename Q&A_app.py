import os
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

def get_secret(name: str, default=None):
    return st.secrets.get(name) or os.getenv(name) or default

# Secrets (Cloud) or env vars (local)
hf_token = get_secret("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

groq_key = get_secret("GROQ_API_KEY")
if not groq_key:
    st.error("Missing GROQ_API_KEY. Add it in Streamlit Cloud Secrets.")
    st.stop()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant")

session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

# Build vectorstore once per upload set
if uploaded_files:
    # a simple fingerprint to detect changes in uploads
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

        st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        st.session_state.upload_sig = upload_sig

    retriever = st.session_state.retriever

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt),
         MessagesPlaceholder("chat_history"),
         ("human", "{input}")]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt),
         MessagesPlaceholder("chat_history"),
         ("human", "{input}")]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("Assistant:", response["answer"])
        st.write("Chat History:", session_history.messages)
else:
    st.info("Upload at least one PDF to start.")
