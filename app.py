import json
import re
import time
from datetime import datetime
from pathlib import Path
import uuid

import requests
import streamlit as st


HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BASE_DIR = Path(__file__).resolve().parent
CHATS_DIR = BASE_DIR / "chats"
MEMORY_FILE = BASE_DIR / "memory.json"
STREAM_DELAY_SECONDS = 0.02


def load_hf_token() -> str | None:
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return None

    if not token or not token.strip():
        return None
    return token.strip()


def default_memory() -> dict:
    return {
        "name": None,
        "communication_style": [],
        "interests": [],
    }


def ensure_memory_file() -> None:
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(json.dumps(default_memory(), indent=2), encoding="utf-8")


def normalize_memory(memory: dict) -> dict:
    normalized = default_memory()
    if not isinstance(memory, dict):
        return normalized

    for key, default_value in normalized.items():
        value = memory.get(key, default_value)
        if isinstance(default_value, list):
            cleaned_items: list[str] = []
            raw_items: list[str] = []

            if isinstance(value, list):
                raw_items = [item for item in value if isinstance(item, str)]
            elif isinstance(value, str) and value.strip():
                split_parts = re.split(r",|\band\b|\bor\b|;|\n", value)
                raw_items = [part for part in split_parts if isinstance(part, str)]

            for item in raw_items:
                cleaned_item = item.strip(" .,!?:;\"'")
                if cleaned_item:
                    cleaned_items.append(cleaned_item)

            normalized[key] = cleaned_items
        else:
            normalized[key] = value.strip() if isinstance(value, str) and value.strip() else None
    return normalized


def load_memory() -> dict:
    ensure_memory_file()
    try:
        memory = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        memory = default_memory()
    return sanitize_extracted_memory(memory)


def save_memory(memory: dict) -> None:
    MEMORY_FILE.write_text(json.dumps(sanitize_extracted_memory(memory), indent=2), encoding="utf-8")


def merge_memory(existing: dict, extracted: dict) -> dict:
    merged = sanitize_extracted_memory(existing)
    incoming = sanitize_extracted_memory(extracted)

    for key in ("name",):
        if incoming[key]:
            merged[key] = incoming[key]

    for key in ("communication_style", "interests"):
        seen = {item.lower(): item for item in merged[key]}
        for item in incoming[key]:
            lowered = item.lower()
            if lowered not in seen:
                seen[lowered] = item
        merged[key] = list(seen.values())

    return merged


def build_memory_system_message(memory: dict) -> dict | None:
    memory = normalize_memory(memory)
    if not any(
        [
            memory["name"],
            memory["communication_style"],
            memory["interests"],
        ]
    ):
        return None

    memory_lines = [
        "You are chatting with a returning user.",
        "The stored user memory below is authoritative for personalization.",
        "Use it proactively to make responses feel tailored, natural, and consistent across conversations.",
        "If the user asks about their name, interests, or communication preferences, answer from this memory when available.",
        "When giving examples, explanations, recommendations, or follow-up questions, prefer details that connect to the stored interests and communication style.",
        "If a communication style is stored, follow it as a high-priority instruction for tone, structure, and length.",
        "If the style includes concise or brief, keep responses short, direct, and low on extra explanation unless the user asks for more detail.",
        "If the style includes simple, use plain language and avoid unnecessary complexity.",
        "If the style includes detailed, provide more explanation and context.",
        "If the style includes formal or casual, adapt the tone accordingly.",
        "If a name is stored, use it naturally when appropriate, especially in directly personal replies.",
        "Do not invent personal facts that are not present in this memory.",
        "Stored user memory:",
    ]
    if memory["name"]:
        memory_lines.append(f"- Name: {memory['name']}")
    if memory["communication_style"]:
        memory_lines.append(f"- Communication style: {', '.join(memory['communication_style'])}")
    if memory["interests"]:
        memory_lines.append(f"- Interests: {', '.join(memory['interests'])}")

    return {"role": "system", "content": "\n".join(memory_lines)}


def sanitize_extracted_memory(memory: dict) -> dict:
    cleaned = normalize_memory(memory)
    invalid_name_values = {
        "user",
        "the user",
        "john",
        "jane",
        "john doe",
        "jane doe",
        "unknown",
        "none",
        "n/a",
        "not provided",
        "not specified",
    }

    if cleaned["name"] and cleaned["name"].strip().lower() in invalid_name_values:
        cleaned["name"] = None

    return cleaned


def extract_name_from_user_message(user_message: str) -> str | None:
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z' -]{0,40})",
        r"\bcall me\s+([A-Za-z][A-Za-z' -]{0,40})",
        r"\bname's\s+([A-Za-z][A-Za-z' -]{0,40})",
    ]
    invalid_name_values = {
        "user",
        "the user",
        "john",
        "jane",
        "john doe",
        "jane doe",
        "unknown",
        "none",
        "n/a",
        "not provided",
        "not specified",
    }

    for pattern in patterns:
        match = re.search(pattern, user_message, flags=re.IGNORECASE)
        if not match:
            continue

        candidate = match.group(1).strip(" .,!?:;\"'")
        lowered = candidate.lower()
        if lowered in invalid_name_values:
            continue

        words = [word for word in candidate.split() if word]
        if not words:
            continue

        normalized_words = [word[0].upper() + word[1:] if len(word) > 1 else word.upper() for word in words]
        return " ".join(normalized_words)

    return None


def message_explicitly_introduces_name(user_message: str) -> bool:
    introduction_patterns = [
        r"\bmy name is\b",
        r"\bcall me\b",
        r"\bname's\b",
    ]
    return any(re.search(pattern, user_message, flags=re.IGNORECASE) for pattern in introduction_patterns)


def split_preference_items(text: str) -> list[str]:
    parts = re.split(r",|\band\b|\bor\b|;|\n", text)
    cleaned: list[str] = []
    for part in parts:
        item = part.strip(" .,!?:;\"'")
        if item:
            cleaned.append(item)
    return cleaned


def first_sentence_fragment(text: str) -> str:
    parts = re.split(r"[.!?](?:\s|$)", text, maxsplit=1)
    return parts[0].strip()


def extract_rule_based_memory(user_message: str) -> dict:
    extracted = default_memory()
    lowered_message = user_message.lower()

    direct_name = extract_name_from_user_message(user_message)
    if direct_name:
        extracted["name"] = direct_name

    interest_patterns = [
        r"\bi like (.+)",
        r"\bi love (.+)",
        r"\bi enjoy (.+)",
        r"\bi am interested in (.+)",
        r"\bmy interests are (.+)",
    ]
    for pattern in interest_patterns:
        match = re.search(pattern, user_message, flags=re.IGNORECASE)
        if match:
            extracted["interests"] = split_preference_items(first_sentence_fragment(match.group(1)))
            break

    if "concise" in lowered_message:
        extracted["communication_style"].append("concise")
    if "brief" in lowered_message:
        extracted["communication_style"].append("brief")
    if "detailed" in lowered_message:
        extracted["communication_style"].append("detailed")
    if "simple" in lowered_message:
        extracted["communication_style"].append("simple")
    if "formal" in lowered_message:
        extracted["communication_style"].append("formal")
    if "casual" in lowered_message:
        extracted["communication_style"].append("casual")

    return sanitize_extracted_memory(extracted)


def stream_chat_reply(hf_token: str, messages: list[dict[str, str]]):
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "stream": True,
    }

    yielded_any = False

    with requests.post(
        HF_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
        stream=True,
    ) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue

            data = raw_line[6:].strip()
            if data == "[DONE]":
                break

            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = parsed.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if not content:
                continue

            yielded_any = True
            yield content
            time.sleep(STREAM_DELAY_SECONDS)

    if not yielded_any:
        raise ValueError("The streamed API response did not include any message content.")


def extract_memory(hf_token: str, user_message: str) -> dict:
    headers = {"Authorization": f"Bearer {hf_token}"}
    extraction_prompt = (
        "Extract persistent user memory from this message. "
        "Return JSON only, with no markdown, explanation, or surrounding text. "
        "Capture only facts the user explicitly states about themself, such as their name, interests, or communication preferences. "
        "Do not guess or infer missing facts. "
        "Never use placeholders like 'user', 'the user', 'unknown', or 'not provided' as values. "
        "Return only valid JSON using exactly these keys: "
        "name, communication_style, interests. "
        "Use null for missing string values and [] for missing list values. "
        "If the message contains no stable personal facts or preferences, return {}.\n\n"
        f"User message: {user_message}"
    )
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": extraction_prompt}],
        "max_tokens": 256,
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("The memory extraction response did not include any choices.")

    content = choices[0].get("message", {}).get("content", "").strip()
    if not content:
        return {}

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The memory extraction response was not valid JSON.")

    parsed_memory = json.loads(content[start : end + 1])
    if not isinstance(parsed_memory, dict):
        raise ValueError("The memory extraction response was not a JSON object.")
    parsed_memory = sanitize_extracted_memory(parsed_memory)
    fallback_memory = extract_rule_based_memory(user_message)
    parsed_memory = merge_memory(parsed_memory, fallback_memory)

    direct_name = extract_name_from_user_message(user_message)
    if direct_name:
        parsed_memory["name"] = direct_name
    elif parsed_memory.get("name") and not message_explicitly_introduces_name(user_message):
        parsed_memory["name"] = None

    return sanitize_extracted_memory(parsed_memory)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_chat_title(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user":
            content = " ".join(message["content"].split()).strip()
            if content:
                return content[:40] + ("..." if len(content) > 40 else "")
    return "New Chat"


def build_chat() -> dict:
    timestamp = now_timestamp()
    return {
        "id": uuid.uuid4().hex,
        "title": "New Chat",
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }


def ensure_chats_dir() -> None:
    CHATS_DIR.mkdir(exist_ok=True)


def chat_file_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def sort_chats(chats: list[dict]) -> list[dict]:
    return sorted(chats, key=lambda chat: chat.get("updated_at", ""), reverse=True)


def is_valid_chat(chat: dict) -> bool:
    required_keys = {"id", "title", "created_at", "updated_at", "messages"}
    return isinstance(chat, dict) and required_keys.issubset(chat.keys()) and isinstance(chat["messages"], list)


def save_chat(chat: dict) -> None:
    ensure_chats_dir()
    chat_file_path(chat["id"]).write_text(json.dumps(chat, indent=2), encoding="utf-8")


def load_chats() -> list[dict]:
    ensure_chats_dir()
    loaded_chats: list[dict] = []

    for chat_file in CHATS_DIR.glob("*.json"):
        try:
            chat = json.loads(chat_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if is_valid_chat(chat):
            loaded_chats.append(chat)

    return sort_chats(loaded_chats)


def delete_chat_file(chat_id: str) -> None:
    file_path = chat_file_path(chat_id)
    if file_path.exists():
        file_path.unlink()


def initialize_session_state() -> None:
    if "chats" not in st.session_state:
        loaded_chats = load_chats()
        if loaded_chats:
            st.session_state.chats = loaded_chats
            st.session_state.active_chat_id = loaded_chats[0]["id"]
        else:
            first_chat = build_chat()
            save_chat(first_chat)
            st.session_state.chats = [first_chat]
            st.session_state.active_chat_id = first_chat["id"]

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"] if st.session_state.chats else None

    if "memory_status" not in st.session_state:
        st.session_state.memory_status = ""


def get_chat_index(chat_id: str) -> int | None:
    for index, chat in enumerate(st.session_state.chats):
        if chat["id"] == chat_id:
            return index
    return None


def get_active_chat() -> dict | None:
    active_chat_id = st.session_state.active_chat_id
    if not active_chat_id:
        return None

    chat_index = get_chat_index(active_chat_id)
    if chat_index is None:
        return None
    return st.session_state.chats[chat_index]


def create_new_chat() -> None:
    new_chat = build_chat()
    save_chat(new_chat)
    st.session_state.chats.insert(0, new_chat)
    st.session_state.chats = sort_chats(st.session_state.chats)
    st.session_state.active_chat_id = new_chat["id"]


def set_active_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id


def delete_chat(chat_id: str) -> None:
    remaining_chats = [chat for chat in st.session_state.chats if chat["id"] != chat_id]
    delete_chat_file(chat_id)
    st.session_state.chats = sort_chats(remaining_chats)

    if st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"] if st.session_state.chats else None


def update_chat(chat_id: str, messages: list[dict[str, str]]) -> None:
    chat_index = get_chat_index(chat_id)
    if chat_index is None:
        return

    chat = st.session_state.chats[chat_index]
    chat["messages"] = messages
    chat["title"] = make_chat_title(messages)
    chat["updated_at"] = now_timestamp()
    save_chat(chat)
    st.session_state.chats = sort_chats(st.session_state.chats)


st.set_page_config(page_title="My AI Chat", layout="wide")
initialize_session_state()
memory = load_memory()

with st.sidebar:
    st.subheader("Chats")
    if st.button("New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.divider()

    with st.expander("User Memory", expanded=True):
        st.json(memory)
        if st.button("Clear Memory", use_container_width=True):
            save_memory(default_memory())
            st.session_state.memory_status = "Memory cleared."
            st.rerun()
        if st.session_state.memory_status:
            st.caption(st.session_state.memory_status)

    st.divider()

    if st.session_state.chats:
        for chat in st.session_state.chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            label = f"{chat['title']}\n{chat['updated_at']}"

            button_col, delete_col = st.columns([5, 1])
            if button_col.button(
                label,
                key=f"chat-select-{chat['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                set_active_chat(chat["id"])
                st.rerun()

            if delete_col.button("\u2715", key=f"chat-delete-{chat['id']}", use_container_width=True):
                delete_chat(chat["id"])
                st.rerun()
    else:
        st.caption("No chats yet. Create one to get started.")

st.title("My AI Chat")
st.write("Ask a question to start a multi-turn conversation.")

hf_token = load_hf_token()
if not hf_token:
    st.error("Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` and rerun the app.")

active_chat = get_active_chat()
if active_chat is None and st.session_state.chats:
    st.session_state.active_chat_id = st.session_state.chats[0]["id"]
    active_chat = get_active_chat()

if active_chat is None:
    st.info("No active chat. Create a new chat from the sidebar.")
else:
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Type your message here", disabled=active_chat is None)

if prompt and active_chat is not None:
    user_message = {"role": "user", "content": prompt}
    active_chat["messages"].append(user_message)
    update_chat(active_chat["id"], active_chat["messages"])
    active_chat = get_active_chat()

    with st.chat_message("user"):
        st.markdown(prompt)

    if not hf_token:
        with st.chat_message("assistant"):
            st.error("Cannot send messages without a valid Hugging Face token in `.streamlit/secrets.toml`.")
    else:
        memory_system_message = build_memory_system_message(memory)
        messages_for_model = active_chat["messages"][:]
        if memory_system_message is not None:
            messages_for_model = [memory_system_message] + messages_for_model

        with st.chat_message("assistant"):
            try:
                streamed_reply = st.write_stream(stream_chat_reply(hf_token, messages_for_model))
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else "unknown"
                error_text = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Hugging Face API error ({status_code}): {error_text}")
            except requests.RequestException as exc:
                st.error(f"Network error while contacting Hugging Face: {exc}")
            except ValueError as exc:
                st.error(f"Unexpected streamed response: {exc}")
            else:
                assistant_message = {"role": "assistant", "content": streamed_reply}
                active_chat["messages"].append(assistant_message)
                update_chat(active_chat["id"], active_chat["messages"])

                try:
                    extracted_memory = extract_memory(hf_token, prompt)
                except (requests.RequestException, ValueError, json.JSONDecodeError):
                    fallback_memory = extract_rule_based_memory(prompt)
                    if any(
                        [
                            fallback_memory["name"],
                            fallback_memory["communication_style"],
                            fallback_memory["interests"],
                        ]
                    ):
                        merged_memory = merge_memory(memory, fallback_memory)
                        save_memory(merged_memory)
                        st.session_state.memory_status = "Memory updated from local extraction fallback."
                    else:
                        st.session_state.memory_status = "Memory extraction skipped due to a non-fatal error."
                else:
                    if extracted_memory:
                        merged_memory = merge_memory(memory, extracted_memory)
                        save_memory(merged_memory)
                        st.session_state.memory_status = "Memory updated."
                    else:
                        st.session_state.memory_status = "No new memory extracted."
