import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os


# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# ----------------------------
# Per-user Groq API key (stable across reruns)
# ----------------------------
with st.sidebar:
    if "user_groq_key" not in st.session_state:
        st.session_state.user_groq_key = ""

    groq_api_key = st.text_input("Groq API Key", type="password", key="user_groq_key")

# sanitize
groq_api_key = (groq_api_key or "").strip()

# Optional local fallback to keep your local workflow intact
if not groq_api_key:
    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()

# URL input
generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="Paste a YouTube or website URL here...")
generic_url = (generic_url or "").strip()

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not groq_api_key or not generic_url:
        st.error("Please provide the Groq API key and a URL to get started.")
        st.stop()

    if not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube link or website URL).")
        st.stop()

    try:
        with st.spinner("Waiting..."):
            # Create LLM only after key validation
            llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)

            # Load the website or YT video data
            if "youtube.com" in generic_url or "youtu.be" in generic_url:
                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
            else:
                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/116.0.0.0 Safari/537.36"
                        )
                    },
                )

            docs = loader.load()

            # Chain for summarization
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            output_summary = chain.run(docs)

            st.success(output_summary)

    except Exception as e:
        st.exception(e)
