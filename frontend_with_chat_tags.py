import streamlit as st
from backend import chatbot, retrieve_threads
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import uuid

from dotenv import load_dotenv
load_dotenv()

def _extract_text(content):
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    # Some providers return a list of dicts / parts
    try:
        if isinstance(content, list):
            parts = []
            for part in content:
                # OpenAI style: {'type':'text','text': '...'} or just {'text': '...'}
                if isinstance(part, dict):
                    txt = part.get('text') or part.get('content') or ""
                    if txt:
                        parts.append(str(txt))
                else:
                    parts.append(str(part))
            return "\n".join(p.strip() for p in parts if p and p.strip())
    except Exception:
        return str(content)
    return str(content).strip()

def add_thread_to_threads(thread_id):
    if thread_id not in st.session_state['threads']:
        st.session_state['threads'].append(thread_id)

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_history():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['history'] = []

def load_conversation(thread_id):
    # state = chatbot.get_state(config = {'configurable': {'thread_id': thread_id}}).values['messages']
    # print(state.values)
   
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])
    return messages

@st.cache_data(show_spinner=False)
def get_thread_preview(thread_id: str) -> str:
    try:
        messages = load_conversation(thread_id)
        for msg in messages:
            if isinstance(msg, HumanMessage):
                line = msg.content.strip().splitlines()[0]
                if len(line) > 60:
                    line = line[:57] + '...'
                return line or "(empty)"
        return "(no user message yet)"
    except Exception:
        return "(unavailable)"

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'threads' not in st.session_state:
    st.session_state['threads'] = retrieve_threads()


if 'thread_id' not in st.session_state:
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id

add_thread_to_threads(st.session_state['thread_id'])

# CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
CONFIG = {
    'configurable': {'thread_id': st.session_state['thread_id']},
    'metadata' : {'thread_id': st.session_state['thread_id']},
    'run_name' : 'chat_run'  
}
#sidebar
st.sidebar.title('Chatbot')
if st.sidebar.button('new chat'):
    reset_history()
    add_thread_to_threads(st.session_state['thread_id'])

st.sidebar.header('previous chats')
for thread_id in st.session_state['threads']:
    label = get_thread_preview(thread_id)
    if st.sidebar.button(label, key=f"thread_btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        all_msgs = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                all_msgs.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                text = _extract_text(msg.content)
                if text:
                    all_msgs.append({'role': 'ai', 'content': text})
            else:
                continue
        st.session_state['history'] = all_msgs

#flow

for message in st.session_state['history']:
    if isinstance(message['content'], str):
        with st.chat_message(message['role']):
            st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    is_first_user = not any(m['role'] == 'user' for m in st.session_state['history'])
    st.session_state['history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    ai_state = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)
    # Find last AI message with non-empty textual content
    final_ai_content = ""
    for msg in reversed(ai_state['messages']):
        if isinstance(msg, AIMessage):
            txt = _extract_text(msg.content)
            if txt:
                final_ai_content = txt
                break
    if not final_ai_content:
        final_ai_content = "(No textual response)"
    st.session_state['history'].append({'role': 'ai', 'content': final_ai_content})
    with st.chat_message('ai'):
        st.text(final_ai_content)

    # Invalidate cached previews so sidebar updates for brand new first prompt
    if is_first_user:
        get_thread_preview.clear()