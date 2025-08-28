# Chatbot LangGraph

A conversational AI assistant with Streamlit/Gradio UIs, built with LangGraph + LangChain. Supports tool calling (web search, PDF RAG), stateful threads (SQLite), and dynamic PDF indexing.

## Features
- **LangGraph**: `chat_node` for LLM responses, `tools` node for tool execution.
- **Tools**: DuckDuckGo search, PDF-based RAG (`rag_tool`).
- **RAG**: Auto-builds FAISS index for PDFs in `docs/`; reuses unless files change.
- **Persistence**: SQLite (`SqliteSaver`) for thread isolation and resumption.
- **UI**: Streamlit (`frontend_with_chat_tags.py`) with thread switching; optional Gradio.

## Folder Structure
```
chatbot_langgraph/
├─ backend.py                # Graph, tools, RAG
├─ frontend_with_chat_tags.py# Streamlit UI
├─ requirements.txt          # Dependencies
├─ docs/                     # PDF knowledge base
├─ chatbot.db*               # SQLite checkpoints
├─ faiss_index/              # FAISS index
├─ index_meta.pkl            # Index metadata
```

## Prerequisites
- Python 3.10+
- OpenAI API key
- (Optional) Internet for DuckDuckGo search

## Environment Setup
Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

## Installation
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows; use `source .venv/bin/activate` for Unix
pip install -r requirements.txt
```

## Adding Knowledge (RAG)
1. Place PDFs in `docs/`.
2. First query builds FAISS index (`chunk_size=500`, `chunk_overlap=100`).
3. Delete `faiss_index/` and `index_meta.pkl` to force rebuild if needed.

## Running the App
```bash
streamlit run frontend_with_chat_tags.py
```
Access the local URL. Use the sidebar to start a new chat or switch threads (labeled by first user prompt).

## How It Works
A `StateGraph` orchestrates:
- **chat_node**: Runs `ChatOpenAI` with tools; formats messages.
- **tools**: Executes DuckDuckGo search or RAG if triggered by `tools_condition`.
- **RAG**: Retrieves PDF chunks via FAISS for augmented prompts.
- **Persistence**: `SqliteSaver` stores thread state with unique `thread_id`.

## Extending Tools
Add a new tool:
```python
from langchain.tools import tool

@tool
def my_internal_calc(expression: str) -> str:
    """Safely evaluate a math expression."""
    return str(eval(expression))  # Add safety checks
tools.append(my_internal_calc)
llm_with_tools = model.bind_tools(tools)
```
Update the `chat_node` system prompt to include the new tool.

## 🤝 Contributing
Feel free to fork this repo, open issues, or submit pull requests!

## 👨‍💻 Author & Connect
Created by [Naimah](https://github.com/icebeartellsnolies). Reach out via [GitHub Issues](https://github.com/icebeartellsnolies/RAGraphChat/issues) or [LinkedIn](https://www.linkedin.com/in/naimah-rehman-b1390b283/).