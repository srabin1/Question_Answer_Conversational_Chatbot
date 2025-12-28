import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_classic.sql_database import SQLDatabase
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import os

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLLite 3 Database- Student.db", "Connect to your MySQL Database"]
selected_opt = st.sidebar.radio(label="Choose the DB which you want to chat", options=radio_opt)

# use sqlite is index=0; mysql is index=1
if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MYSQL User")
    mysql_password = st.sidebar.text_input("MYSQL password", type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_uri = LOCALDB
    mysql_host = mysql_user = mysql_password = mysql_db = None

# ----------------------------
# Per-user Groq API key (stored in session_state)
# ----------------------------
if "user_groq_key" not in st.session_state:
    st.session_state.user_groq_key = ""

api_key = st.sidebar.text_input("Groq API Key", type="password", key="user_groq_key")
api_key = (api_key or "").strip()

# Optional local fallback so your local machine can still use env var
if not api_key:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()

if not api_key:
    st.info("Please add the Groq API key in the sidebar.")
    st.stop()

# ----------------------------
# LLM model (IMPORTANT: use api_key, not env-only key)
# ----------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    streaming=True,
    temperature=0,
    model_kwargs={"tool_choice": "auto"},
)

@st.cache_resource(ttl=2 * 60 * 60)  # 2 hours
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))

    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(
            create_engine(
                f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            )
        )

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

# toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        result = agent.invoke({"input": user_query}, config={"callbacks": [streamlit_callback]})
        response = result["output"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
