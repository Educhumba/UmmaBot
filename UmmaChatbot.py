import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
app = FastAPI()
# Allow frontend to access from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use specific origin like ["https://umma.co.ke"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(query: Query):
    reply = smart_chat(query.question)
    return {"response": reply}
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# 1️⃣ Load PDF
loader = PyPDFLoader("Umma_Insurance_Training_Document.pdf")
docs = loader.load()

# 2️⃣ Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3️⃣ Create or load Chroma DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if os.path.exists("db_groq"):
    vectordb = Chroma(persist_directory="db_groq", embedding_function=embeddings)
else:
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db_groq")
    vectordb.persist()

# 4️⃣ Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-8b-8192",
    temperature=0.3,
    max_tokens=512
)

# 5️⃣ Build QA chain
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are UmmaBot, an insurance assistant chatbot for Umma Insurance. Answer the user's question using information from the provided context. Be concise, clear, and human-like.
- Avoid repeating company slogans or marketing language.
- Do NOT hallucinate information not in the context.
- If the context is unclear, politely say so.
- Format step-by-step answers as numbered or bulleted lists, each on a new line.
- Keep responses under 100 words unless absolutely necessary.
---------------------
{context}
---------------------
User Question: {question}
Your Response:"""
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
)

# 6️⃣ Define chatbot logic
# Global memory variable to track chat flow
chat_history = []
def smart_chat(query):
    if not query.strip():
        return "Please enter a valid question."
    try:
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})
        chat_history.append((query, result["answer"]))  # Storing the conversation manually

        if not result.get("source_documents"):
            return "Sorry, I couldn't find anything related to that in Umma Insurance's materials."
        
        return result["answer"].strip()
    except Exception as e:
        print("ERROR :", str(e))
        return f"An error occurred: {str(e)}"