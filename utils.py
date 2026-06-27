import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

def get_vector_store():
    if not os.path.exists("chroma_db"):
        raise FileNotFoundError("Database 'chroma_db' not found. Run ingest.py first.")
        
    return Chroma(
        persist_directory="chroma_db", 
        embedding_function=MistralAIEmbeddings()
    )

def answer_question(question: str):
    db = get_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the provided context.
If the answer is not found in the context, respond exactly with:
"I couldn't find that information in the provided document."
Do not make up information.

Context:
{context}

Question: {question}

Answer:
""")
    
    chain = prompt | llm
    res = chain.invoke({"context": context, "question": question})
    
    return res.content, docs
