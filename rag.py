from dotenv import load_dotenv
import os
import ssl
import certifi
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_tavily import TavilySearch, TavilyCrawl, TavilyMap, TavilyExtract




from langchain_core.prompts import (
    PromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
)

from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from tenacity import retry, stop_after_attempt, wait_fixed


import operator
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pprint import pprint


# variables
from variables import elon_musk_info, summary_template
import logging
import json

load_dotenv()

ssl_context = ssl.create_default_context(cafile = certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth = 5, max_breadth = 20, max_pages = 1000)
tavily_crawl = TavilyCrawl()



# ------------------------------------------------------------IMPORTS------------------------------------------------------------


llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    # other params...
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")


# Reusable retry decorator for robustness
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def safe_invoke(llm, messages):
    return llm.invoke(messages)


def create_the_store(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

        # # print(text)
        # chunk_size = 200
        # docs = []
        # for i in range(0, len(text), chunk_size):
        #     chunk = text[i:i + chunk_size]
        #     doc = Document(page_content=chunk, metadata={"chunk_id": i // chunk_size})
        #     docs.append(doc)
        # print(docs)
        # print("`" * 100)

        docs = [Document(page_content=text)]

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        print(chunks[:2])

        # 5. Create FAISS vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("vector_index")


def similarity_search(query: str, k: int = 4):
    # Load the FAISS vector store
    vectorstore = FAISS.load_local(
        "web_crawler", embeddings, allow_dangerous_deserialization=True
    )
    results = vectorstore.similarity_search(query, k=k)
    print(results)
    print("`" * 200)
    
    # # Define your threshold (0.7 = good starting point)
    # score_threshold = 0.5

    # # Filter results based on similarity score
    # filtered_results = [
    #     (doc, score) for doc, score in results if score >= score_threshold
    # ]
    return results


def rag_app():
    print("RAG application started.")
    # create_the_store('star_wars.txt')
    question = " what are the varius options to avoid crawling duplicate URLs ? "
    results = similarity_search(question , k=10)
    
    context = ""
    for res in results:
        context += res.page_content + "\n"

    print(context)
    print("`" * 200)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an intelligent assistant that answers questions strictly based on the provided context.\n"
                    "here is the context => {context}"
                    "Use only the given information to answer. If the context does not contain the answer, "
                    "say 'The information is not available in the context.'\n"
                    "Keep your answer concise, factual, and clear."
                ),
            ),
            ("user", "Question:{question}"),
        ]
    )
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    print(response)
def clean_contentf(text):
    lines = text.splitlines()
    cleaned = [
        line.strip() for line in lines
        if line.strip()
        and not line.startswith("* [")
        and not line.startswith("[My Account]")
        and not "javascript:void" in line
    ]
    return "\n".join(cleaned)
def crawl_the_site():
    print("Hello world")
    # results = tavily_crawl.invoke({"url": "https://www.starwars.com", "max_depth": 2})
    # for r in results['results'][:1]:
    #     print("🔗", r["url"])
    #     print(r["raw_content"][:200])
    #     print("-" * 80)
    
    urls = [
        "https://www.starwars.com/news",
    ]

    clean_text = tavily_extract.run({"urls": ["https://blog.bytebytego.com/p/how-to-avoid-crawling-duplicate-urls"]})
    raw_content = clean_text['results'][0]['raw_content']
    clean_content = clean_contentf(raw_content)
    docs = [Document(page_content=clean_content)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 5. Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("web_crawler")
    print("-" * 80)


if __name__ == "__main__":
    print(os.environ.get("OPENAI_API_KEY"))
    rag_app()
    # crawl_the_site()
