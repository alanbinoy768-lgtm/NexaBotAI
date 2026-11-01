from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from src.config import DEFAULT_OLLAMA_MODEL, MAX_USER_MESSAGE_CHARS, ensure_directories
from src.database import ChatDatabase
from src.ollama_client import OllamaClient, OllamaError


st.set_page_config(page_title="NexaBot AI", page_icon="🤖", layout="wide")

client = OllamaClient()


def create_new_conversation(db: ChatDatabase) -> None:
    title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    st.session_state["current_conversation_id"] = db.create_conversation(title)
    st.session_state["messages"] = []
    st.session_state.pop("conversation_selector", None)


def load_conversation(db: ChatDatabase, conversation_id: int) -> None:
    st.session_state["current_conversation_id"] = conversation_id
    st.session_state["messages"] = [
        {
            "sender": message["sender"],
            "text": message["message_text"],
            "model_used": message.get("model_used"),
        }
        for message in db.list_messages(conversation_id)
    ]


def to_ollama_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "role": "assistant" if message["sender"] == "bot" else "user",
            "content": message["text"],
        }
        for message in messages
        if message.get("sender") in {"user", "bot"}
    ]


def get_connection_status() -> Dict[str, Any]:
    try:
        return client.check_connection()
    except OllamaError as error:
        return {"available": False, "models": [], "error": str(error)}


ensure_directories()
db = ChatDatabase()
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_conversation_id" not in st.session_state:
    st.session_state["current_conversation_id"] = None
if st.session_state["current_conversation_id"] is None:
    create_new_conversation(db)

status = get_connection_status()
available_models = status["models"]
model_options = list(available_models)
if DEFAULT_OLLAMA_MODEL not in model_options:
    model_options.insert(0, DEFAULT_OLLAMA_MODEL)
if "selected_ollama_model" not in st.session_state:
    st.session_state["selected_ollama_model"] = (
        DEFAULT_OLLAMA_MODEL
        if DEFAULT_OLLAMA_MODEL in model_options
        else model_options[0]
    )
if st.session_state["selected_ollama_model"] not in model_options:
    st.session_state["selected_ollama_model"] = DEFAULT_OLLAMA_MODEL

st.markdown(
    """
    <style>
        .stApp { background: #0a1220; color: #eef4ff; }
        .block-container { padding-top: 1.5rem; }
        [data-testid="stChatMessage"] {
            border-radius: 18px;
            border: 1px solid #263449;
            background: #111b2b;
        }
        .sidebar .stButton > button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("NexaBot AI")
st.sidebar.caption("Your Intelligent Local AI Assistant")

if st.sidebar.button("New conversation", use_container_width=True):
    create_new_conversation(db)
    st.rerun()

conversation_list = db.list_conversations()
conversation_ids = [item["id"] for item in conversation_list]
if conversation_ids:
    current_index = (
        conversation_ids.index(st.session_state["current_conversation_id"])
        if st.session_state["current_conversation_id"] in conversation_ids
        else 0
    )
    selected_conversation = st.sidebar.selectbox(
        "Chat history",
        options=conversation_ids,
        index=current_index,
        key="conversation_selector",
        format_func=lambda conversation_id: next(
            (
                item["title"]
                for item in conversation_list
                if item["id"] == conversation_id
            ),
            str(conversation_id),
        ),
    )
    if selected_conversation != st.session_state["current_conversation_id"]:
        load_conversation(db, selected_conversation)
        st.rerun()

clear_col, delete_col = st.sidebar.columns(2)
if clear_col.button("Clear chat", use_container_width=True):
    db.clear_conversation(st.session_state["current_conversation_id"])
    st.session_state["messages"] = []
    st.rerun()
if delete_col.button("Delete chat", use_container_width=True):
    db.delete_conversation(st.session_state["current_conversation_id"])
    create_new_conversation(db)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Local AI")
if status["available"]:
    st.sidebar.success("Ollama is running locally.")
else:
    st.sidebar.error(status["error"])
    st.sidebar.caption(
        "Install Ollama from ollama.com/download, start it, then restart NexaBot."
    )

selected_model = st.sidebar.selectbox(
    "Model",
    options=model_options,
    index=model_options.index(st.session_state["selected_ollama_model"]),
    key="selected_ollama_model",
)
if status["available"] and selected_model not in available_models:
    st.sidebar.warning(f"{selected_model} is not downloaded yet.")
    st.sidebar.code(f"ollama pull {selected_model}", language="powershell")

answer_style = st.sidebar.selectbox(
    "Response length",
    options=["Short", "Balanced", "Detailed"],
    index=1,
)
temperature = st.sidebar.slider(
    "Creativity (temperature)",
    min_value=0.0,
    max_value=1.5,
    value=0.7,
    step=0.1,
    help="Lower values are more focused. Higher values allow more varied wording.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("About NexaBot")
st.sidebar.write(
    "NexaBot uses a local Ollama language model to generate answers. "
    "Messages stay on this computer; model downloads need internet once."
)
st.sidebar.caption(f"Ollama endpoint: {client.base_url}")

st.title("🤖 NexaBot AI")
st.caption("Your Intelligent NLP Conversation Assistant")
st.write(
    "Ask questions, explore ideas, write and explain code, or continue a conversation."
)
if status["available"]:
    st.success(f"Local AI connected · Model: {selected_model}")
else:
    st.warning(
        "Local AI is unavailable. NexaBot will not substitute canned answers; "
        "start Ollama to generate responses."
    )

if not st.session_state["messages"]:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I am NexaBot. I can explain topics, help with code, solve "
            "problems, and keep context during this conversation."
        )

for message in st.session_state["messages"]:
    role = "assistant" if message["sender"] == "bot" else "user"
    with st.chat_message(role):
        st.markdown(message["text"])
        if role == "assistant" and message.get("model_used"):
            st.caption(f"Generated locally with {message['model_used']}")

can_chat = status["available"] and selected_model in available_models
user_message = st.chat_input(
    "Message NexaBot...",
    disabled=not can_chat,
    max_chars=MAX_USER_MESSAGE_CHARS,
)

if user_message:
    if not user_message.strip():
        st.info("Please enter a message before sending.")
    else:
        conversation_id = st.session_state["current_conversation_id"]
        db.save_message(
            conversation_id,
            "user",
            user_message,
            model_used=selected_model,
        )
        st.session_state["messages"].append(
            {"sender": "user", "text": user_message, "model_used": selected_model}
        )

        with st.chat_message("user"):
            st.markdown(user_message)

        history = client.prepare_messages(
            to_ollama_history(st.session_state["messages"]),
            answer_style=answer_style,
        )
        with st.chat_message("assistant"):
            try:
                with st.spinner(f"Generating locally with {selected_model}..."):
                    answer = st.write_stream(
                        client.stream_chat(
                            history,
                            model=selected_model,
                            temperature=temperature,
                            max_tokens={
                                "Short": 256,
                                "Balanced": 512,
                                "Detailed": 1024,
                            }[answer_style],
                        )
                    )
                db.save_message(
                    conversation_id,
                    "bot",
                    answer,
                    model_used=selected_model,
                )
                st.session_state["messages"].append(
                    {"sender": "bot", "text": answer, "model_used": selected_model}
                )
                st.caption(f"Generated locally with {selected_model}")
            except (OllamaError, ValueError) as error:
                st.error(str(error))
