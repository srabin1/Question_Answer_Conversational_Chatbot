# Question_Answer_Conversational_Chatbot: 🧠 LangChain + Streamlit AI Apps (Cloud-Ready)

This repository contains a collection of **Streamlit applications powered by LangChain** for building AI assistants, including:

- 📄 RAG (Retrieval-Augmented Generation) over PDFs  
- 🔎 Tool-using agents (Web search, Wikipedia, arXiv)  
- 🗄️ Natural-language querying of SQL databases  
- 🎥🌐 Summarization of YouTube videos and web pages  
- 💬 General Q&A chatbots  

All apps are **Streamlit Cloud–compatible** and designed so that **each user provides their own API key**, ensuring:

- 🔐 No shared credentials  
- 💰 No token costs for the repository owner  
- 🚀 Safe public deployment  

---

## ✨ Key Features

- Per-user API keys (Groq / OpenAI) via Streamlit sidebar  
- Secure handling of keys using `st.session_state`  
- Support for:
  - Groq LLMs
  - OpenAI chat models
  - HuggingFace embeddings
- Optimized for Streamlit Cloud reruns  
- Minimal dependencies, clean architecture  

---

## 📂 Repository Structure (example)

```text
.
├── app.py                          # Main Streamlit app (or multiple apps)
├── student.db                     # SQLite DB for SQL chatbot (if applicable)
├── research_papers/               # PDFs for RAG apps
│   ├── paper1.pdf
│   └── paper2.pdf
├── requirements.txt
└── README.md
```

---
## 🔑 API Keys (How It Works)

All apps are configured so that each user enters their own API key:

- Keys are entered in the Streamlit sidebar
- Keys are stored only in the browser session
- Keys are never logged, saved, or committed

### Supported APIs:

- Groq API key → LLM inference
- OpenAI API key → Chat models or embeddings
- HF_TOKEN (optional) → Improves HuggingFace download limits
