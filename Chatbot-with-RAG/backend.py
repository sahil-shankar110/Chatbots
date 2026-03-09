# LLM Integration
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Embedding Integration
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import streamlit as st

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=None,
        max_retries=2
    )

pdf_path = "ai_ml_foundations.pdf"

@st.cache_resource(show_spinner=False)
def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("✅ Created and saved new FAISS index.")
    
    return vectorstore

llm = get_llm()
vectorstore = create_vector_store()
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3})

def generate_response(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = sorted({int(doc.metadata["page_label"]) for doc in docs if "page_label" in doc.metadata})
    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant with access to the following PDF documents:
    {available_pdfs}

    Context:
    {context}

    Question:
    {question}
    
    Instructions:
    - If the user asks which PDFs, files, or documents you have access to, respond with: "I have access to the following PDF(s):" and list them.
    - If the context is relevant, answer using ONLY the context and be specific and accurate.
    - If the context is NOT relevant or empty, answer using your general knowledge.
    - If answering from general knowledge, start with "Based on my general knowledge:"
    - Never say "I don't know" if you can answer from general knowledge.
    """)

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question,"available_pdfs": os.path.basename(pdf_path)})

    return response, ([] if "Based on my general knowledge:" in response else sources)