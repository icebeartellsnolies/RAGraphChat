from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
# from langchain_core import tools
import hashlib
import pickle
import sqlite3

import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4o-mini')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

index_dir = 'faiss_index'
meta_dir = 'index_meta.pkl'

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def build_or_load(docs_path = "docs", chunk_size=500, chunk_overlap = 100):
    embeddings = OpenAIEmbeddings()
    meta = {'file_path':{}, 'chunk_overlap':chunk_overlap, 'chunk_size':chunk_size}
    
    for file in os.listdir(docs_path):
        if file.endswith('.pdf'):
            file_path = os.path.join(docs_path, file)
            meta['file_path']['file'] = get_file_hash(file_path)

    old_meta = None
    
    if os.path.exists('faiss_index.pkl') and os.path.exists('faiss_meta.pkl'):
        with open('faiss_meta.pkl', 'rb') as f:
            old_meta = pickle.load(f)
            
    if old_meta:
        if old_meta == meta:
            return FAISS.load_local(index_dir, embeddings)
        
    docs = []
    for file in os.listdir(docs_path):
        if file.endswith('.pdf'):
            file_path = os.path.join(docs_path, file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)

    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(index_dir)

    with open(meta_dir, 'wb') as f:
        pickle.dump(meta, f)
    
    return db

search_tool = DuckDuckGoSearchRun(region='us-en')
@tool
def rag_tool(query: str):
    '''perform retrieval augmneted generation(RAG) from file in docs folder if needed'''
    db = build_or_load()
    retriever = db.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)
    content = "\n\n".join(d.page_content for d in relevant_docs)
    prompt = f'''you are a helpful assistant. use the {content} to answer the following question precisely:
    {query}'''
    response = model.invoke([HumanMessage(content=prompt)])
    return response

tools = [search_tool, rag_tool]
tool_node = ToolNode(tools)
llm_with_tools = model.bind_tools(tools)


def chat_node(state: ChatState):
    '''LLM node that may answer or call the tool node if needed'''
    system_prompt = """You are an assistant that can use tools. 
        - DuckDuckGoSearchRun: use for searching the web for general or external information.
        - RAGRetriever: use for retrieving information specifically from the internal knowledge base of documents.

        Guidelines:
        - If the user asks about content that is in the provided documents, company policies, files, or any internal knowledge base, always use RAGRetriever. 
        - If the user asks about general knowledge, news, or things outside the document set, use DuckDuckGoSearchRun.
        - Only answer directly if the question is simple and can be answered without any tool.
        Always prefer the RAG. search among the available documnets first. if you cant find relevant information from the 
        available documents then only perform web search
        Always carefully choose the tool that best matches the source of truth for the user’s request.
    """

    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{messages}")
    ])

    messages = state['messages']
    formatted_prompt = prompt.format_messages(messages=messages)
    response = llm_with_tools.invoke(formatted_prompt)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)