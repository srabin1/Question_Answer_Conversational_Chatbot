import os
import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="LangChain: Summarize Text From YouTube or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YouTube or Website")
st.subheader("Summarize URL")

with st.sidebar:
    st.markdown("### Settings")
    if "user_groq_key" not in st.session_state:
        st.session_state.user_groq_key = ""
    groq_api_key = st.text_input("Groq API Key", type="password", key="user_groq_key")

# sanitize inputs
groq_api_key = (groq_api_key or "").strip()
if not groq_api_key:
    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()

generic_url = st.text_input(
    "URL",
    label_visibility="collapsed",
    placeholder="Paste a YouTube or website URL here..."
)
generic_url = (generic_url or "").strip()

# Optional controls
with st.expander("Advanced options", expanded=False):
    summary_words = st.slider("Target summary length (words)", min_value=100, max_value=600, value=300, step=50)
    model_name = st.text_input("Groq model", value="openai/gpt-oss-120b")
    yt_languages = st.text_input("YouTube transcript languages (comma-separated)", value="en").strip()
    chain_type = st.selectbox("Summarization chain", ["map_reduce", "refine", "stuff"], index=0)

yt_language_list = [x.strip() for x in yt_languages.split(",") if x.strip()]


# ----------------------------
# Prompts
# ----------------------------
prompt_template = """
Provide a clear, well-structured summary of the following content in about {words} words.
If the content is technical, include key terms and main takeaways.
Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text", "words"])


# ----------------------------
# Loaders
# ----------------------------
def is_youtube_url(url: str) -> bool:
    u = url.lower()
    return ("youtube.com" in u) or ("youtu.be" in u)

def load_youtube_docs(url: str, languages: list[str]) -> list:
    """
    Loads transcript docs from YouTube.
    Works only when captions/transcripts are available for the video.
    """
    # Try with preferred languages first (more predictable)
    if languages:
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,          # reduces breakage risk
                language=languages,            # try specific languages first
            )
            docs = loader.load()
            if docs and docs[0].page_content.strip():
                return docs
        except Exception:
            pass

    # Fallback: try without language restriction
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False
    )
    return loader.load()


# ----------------------------
# Run
# ----------------------------
if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not groq_api_key or not generic_url:
        st.error("Please provide the Groq API key and a URL to get started.")
        st.stop()

    if not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube link or website URL).")
        st.stop()

    try:
        with st.spinner("Loading content and summarizing..."):
            # Create LLM only after key validation
            llm = ChatGroq(model=model_name, groq_api_key=groq_api_key)

            # Load docs
            if is_youtube_url(generic_url):
                st.info("YouTube summarization works only if the video has subtitles/captions available.")
                docs = load_youtube_docs(generic_url, yt_language_list)

                # Guard: empty transcript
                if not docs or not docs[0].page_content.strip():
                    st.error(
                        "Could not retrieve a transcript for this YouTube video. "
                        "This usually happens when captions are unavailable/disabled."
                    )
                    st.stop()
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

                if not docs or not docs[0].page_content.strip():
                    st.error("No readable text found on that webpage.")
                    st.stop()

            # Show quick diagnostics (helps debugging)
            total_chars = sum(len(d.page_content or "") for d in docs)
            st.caption(f"Loaded {len(docs)} document chunk(s), ~{total_chars:,} characters.")

            # Choose summarization chain
            # - map_reduce is best for long transcripts/pages
            # - refine is also good
            # - stuff is only safe for short pages/transcripts
            if chain_type == "stuff":
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                result = chain.invoke({"input_documents": docs, "words": summary_words})
                output_summary = result.get("output_text", str(result))

            elif chain_type == "refine":
                chain = load_summarize_chain(
                    llm,
                    chain_type="refine",
                    question_prompt=prompt,   # prompt used for the initial summary step
                    refine_prompt=prompt      # prompt used for refinement steps
                )
                result = chain.invoke({"input_documents": docs, "words": summary_words})
                output_summary = result.get("output_text", str(result))

            else:  # map_reduce
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=prompt,
                    combine_prompt=prompt
                )
                result = chain.invoke({"input_documents": docs, "words": summary_words})
                output_summary = result.get("output_text", str(result))

            st.success(output_summary)

    except Exception as e:
        st.exception(e)
