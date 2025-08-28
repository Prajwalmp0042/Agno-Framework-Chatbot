import os
import json
import uuid
import asyncio
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain_community.llms import Together
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ----- Constants -----
HISTORY_DIR = "chat_sessions"
VECTORSTORE_PATH = "agno_llama_index"
os.makedirs(HISTORY_DIR, exist_ok=True)

# ----- Utility Functions -----
def get_session_file(session_id):
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

def load_session_history(session_id):
    path = get_session_file(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_session_history(session_id, history):
    with open(get_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def get_all_sessions():
    sessions = []
    for file in os.listdir(HISTORY_DIR):
        if file.endswith(".json"):
            session_id = file[:-5]
            history = load_session_history(session_id)
            title = history[0]["user"][:40] + "..." if history else f"Session: {session_id[:8]}"
            sessions.append((f"{title} ({session_id[:8]})", session_id))
    return sessions

# ----- Ensure Event Loop -----
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ----- Load Env -----
load_dotenv()

# ----- Vector Store Cache -----
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ----- Streamlit Page Setup -----
st.set_page_config(page_title="Agno Framework Chatbot", page_icon="🤖")
st.title("🤖 Agno Framework Chat")

# ----- Session Initialization -----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = load_session_history(st.session_state.session_id)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ----- Sidebar -----
with st.sidebar:
    st.title("🧠 Controls")

    # Load all sessions with previews
    sessions_with_titles = get_all_sessions()
    session_titles = ["<Current>"] + [s[0] for s in sessions_with_titles]
    session_lookup = {title: sid for title, sid in sessions_with_titles}

    selected_title = st.selectbox("📂 Select Previous Session", options=session_titles)

    if selected_title != "<Current>":
        selected_session = session_lookup[selected_title]
        st.session_state.session_id = selected_session
        st.session_state.history = load_session_history(selected_session)
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.success(f"🔄 Loaded session: {selected_session[:8]}")

    if st.button("🆕 New Chat", key="new_chat"):
        new_id = str(uuid.uuid4())
        st.session_state.session_id = new_id
        st.session_state.history = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.success("✅ New chat session started.")

    if st.button("🗑️ Clear Current Session", key="clear_session"):
        st.session_state.history = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        save_session_history(st.session_state.session_id, [])
        st.success("🧹 Current session cleared.")

    if st.button("🔁 Reload Vector Store", key="reload_vector"):
        load_vectorstore.clear()
        st.success("🔄 Vector store reloaded.")

# ----- Prompt Template -----
# ----- Prompt Template -----
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial assistant specializing in analyzing and comparing stock market data.

Use the following context and chat history to answer user queries with clarity and structure.

Respond in this format:

1. **Overview**  
   Summarize the question and what companies or stock metrics are involved.

2. **Data Comparison**  
   Compare stock prices, market caps, P/E ratios, growth trends, or any available metric.  
   Use bullet points or a markdown table when appropriate.

3. **Analysis & Insights**  
   Provide key takeaways based on the comparison.  
   Suggest possible reasons or market events influencing the differences.

4. **Visual Format (Optional)**  
   If relevant, display a flowchart or structured decision logic in ASCII.

5. **Conclusion**  
   Offer a concise summary and what the user might consider next.

Context:
{context}

Question:
{question}
"""
)

# ----- Input Field -----
user_input = st.chat_input("Ask something about the Agno Framework...")

# ----- Process Input -----
if user_input:
    with st.spinner("🤖 Thinking..."):

        # Check for "previous question" pattern
        user_question_lower = user_input.strip().lower()
        if "previous question" in user_question_lower or "last question" in user_question_lower:
            previous_question = None
            for msg in reversed(st.session_state.history):
                if "user" in msg:
                    previous_question = msg["user"]
                    break
            if previous_question:
                response = f"🕘 Your previous question was:\n> {previous_question}"
            else:
                response = "❌ I couldn't find a previous question in this session."
        else:
            # LLM and QA chain
            vectorstore = load_vectorstore()
            llm = Together(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.7,
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={"prompt": prompt_template},
            )

            response = qa_chain.run(user_input).strip()

        # Save turn
        chat_turn = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_input,
            "assistant": response
        }

        st.session_state.history.append(chat_turn)
        save_session_history(st.session_state.session_id, st.session_state.history)

# ----- Display Chat -----
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["assistant"])
