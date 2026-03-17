import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List

import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CHATS_DIR = BASE_DIR / "chats"
MEMORY_FILE = BASE_DIR / "memory.json"
LOG_FILE = BASE_DIR / "ai_interaction_log.md"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"


def ensure_project_files() -> None:
    CHATS_DIR.mkdir(exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(json.dumps({"notes": []}, indent=2), encoding="utf-8")
    if not LOG_FILE.exists():
        LOG_FILE.write_text("# AI Interaction Log\n\n", encoding="utf-8")


def load_memory() -> Dict[str, List[str]]:
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return {"notes": []}


def save_memory(memory: Dict[str, List[str]]) -> None:
    MEMORY_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def list_chat_files() -> List[Path]:
    return sorted(CHATS_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def make_chat_title(messages: List[Dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user":
            title = re.sub(r"\s+", " ", message["content"]).strip()
            return title[:40] + ("..." if len(title) > 40 else "")
    return "New Chat"


def create_chat() -> Dict:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_id = uuid.uuid4().hex
    chat = {
        "id": chat_id,
        "title": "New Chat",
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }
    save_chat(chat)
    return chat


def chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def load_chat(chat_id: str) -> Dict:
    return json.loads(chat_path(chat_id).read_text(encoding="utf-8"))


def save_chat(chat: Dict) -> None:
    chat["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat["title"] = make_chat_title(chat["messages"])
    chat_path(chat["id"]).write_text(json.dumps(chat, indent=2), encoding="utf-8")


def append_log(chat_title: str, user_prompt: str, assistant_reply: str, model: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"## {timestamp} | {chat_title}\n"
        f"- Model: `{model}`\n"
        f"- User: {user_prompt}\n"
        f"- Assistant: {assistant_reply}\n\n"
    )
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(entry)


def build_messages(chat_messages: List[Dict[str, str]], memory_notes: List[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if memory_notes:
        memory_block = "\n".join(f"- {note}" for note in memory_notes)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Use the following persistent user memory when it is relevant.\n"
                    f"{memory_block}"
                ),
            }
        )
    messages.extend(chat_messages)
    return messages


def stream_hf_reply(api_token: str, model: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    with requests.post(HF_API_URL, headers=headers, json=payload, stream=True, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = parsed.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content


def initialize_state() -> None:
    if "active_chat_id" not in st.session_state:
        existing_chats = list_chat_files()
        if existing_chats:
            st.session_state.active_chat_id = existing_chats[0].stem
        else:
            st.session_state.active_chat_id = create_chat()["id"]


ensure_project_files()
st.set_page_config(page_title="Week 10 ChatGPT Clone", page_icon="💬", layout="wide")
initialize_state()

st.title("ChatGPT Clone with Streamlit + Hugging Face")
st.caption("Multi-turn chat, streaming responses, saved conversations, and persistent user memory.")

memory = load_memory()
chat = load_chat(st.session_state.active_chat_id)

with st.sidebar:
    st.subheader("Saved Chats")
    if st.button("New Chat", use_container_width=True):
        new_chat = create_chat()
        st.session_state.active_chat_id = new_chat["id"]
        st.rerun()

    for saved_chat_file in list_chat_files():
        saved_chat = json.loads(saved_chat_file.read_text(encoding="utf-8"))
        label = saved_chat.get("title", "Untitled Chat")
        if st.button(label, key=f"chat-{saved_chat['id']}", use_container_width=True):
            st.session_state.active_chat_id = saved_chat["id"]
            st.rerun()

    st.divider()
    st.subheader("User Memory")
    with st.form("memory_form", clear_on_submit=True):
        new_note = st.text_input("Add a memory note")
        submitted = st.form_submit_button("Save Memory", use_container_width=True)
        if submitted and new_note.strip():
            memory["notes"].append(new_note.strip())
            save_memory(memory)
            st.rerun()

    notes = memory.get("notes", [])
    if notes:
        for index, note in enumerate(notes):
            col1, col2 = st.columns([4, 1])
            col1.write(f"- {note}")
            if col2.button("X", key=f"delete-note-{index}"):
                del memory["notes"][index]
                save_memory(memory)
                st.rerun()
    else:
        st.caption("No memory saved yet.")

    st.divider()
    st.subheader("API Settings")
    model_name = st.text_input(
        "Model",
        value=st.secrets.get("HF_MODEL", DEFAULT_MODEL),
        help="Any chat-capable model available through the Hugging Face Inference Router.",
    )
    token_configured = bool(st.secrets.get("HF_API_TOKEN", ""))
    st.write(f"Token loaded: {'Yes' if token_configured else 'No'}")


for message in chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Send a message")
if prompt:
    if not st.secrets.get("HF_API_TOKEN", ""):
        st.error("Add your Hugging Face API key to `.streamlit/secrets.toml` before chatting.")
        st.stop()

    user_message = {"role": "user", "content": prompt}
    chat["messages"].append(user_message)
    save_chat(chat)

    with st.chat_message("user"):
        st.markdown(prompt)

    assembled_messages = build_messages(chat["messages"], memory.get("notes", []))
    assistant_reply = ""

    with st.chat_message("assistant"):
        response_box = st.empty()
        try:
            for chunk in stream_hf_reply(st.secrets["HF_API_TOKEN"], model_name, assembled_messages):
                assistant_reply += chunk
                response_box.markdown(assistant_reply + "▌")
            response_box.markdown(assistant_reply or "_No response returned._")
        except requests.HTTPError as exc:
            error_text = exc.response.text if exc.response is not None else str(exc)
            assistant_reply = f"Request failed: {error_text}"
            response_box.error(assistant_reply)
        except requests.RequestException as exc:
            assistant_reply = f"Network error: {exc}"
            response_box.error(assistant_reply)

    chat["messages"].append({"role": "assistant", "content": assistant_reply})
    save_chat(chat)
    append_log(chat["title"], prompt, assistant_reply, model_name)
