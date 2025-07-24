import os
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import uuid

#  FastAPI Setup 
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Request Model 
class Query(BaseModel):
    question: str
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # default user_id if not sent

#  Load .env 
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

#  Load PDF 
loader = PyPDFLoader("Training_Document.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

#  Chroma Vector Store 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if os.path.exists("db_groq"):
    vectordb = Chroma(persist_directory="db_groq", embedding_function=embeddings)
else:
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db_groq")

#  LLM Setup 
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-8b-8192",
    temperature=0.3,
    max_tokens=512
)

#  Prompt Template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are UmmaBot, an insurance assistant chatbot for Umma Insurance. Answer the user's question using the context below.
- Avoid slogans or repeated phrases.
- Don't hallucinate. If unsure, say so.
- Use numbered or bullet points for steps.
- Keep responses under 100 words unless needed.
-if you have already welcomed the customer STOP repeating the same in every response and stop re introducing yourself in every response.

{context}

User Question: {question}
Your Response:"""
)

#  QA Chains 
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

general_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever()
)

#  Session Tracking 
user_sessions = {}

#  Google Sheets Logger 
def log_to_sheet(user_query: str, bot_response: str):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("UmmaBot_service_key.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("UmmaBot Logs").sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, user_query, bot_response])
    except Exception as e:
        print("[Logging Error]:", e)

#  Core Logic 
def smart_chat(query: str, user_id: str):
    if not query.strip():
        return "Please enter a valid question."
    try:
        # Maintain context per user
        if user_id not in user_sessions:
            user_sessions[user_id] = []

        history = user_sessions[user_id]
        result = qa_chain.invoke({"question": query, "chat_history": history})
        answer = result.get("answer", "").strip()

        # Store for context
        user_sessions[user_id].append((query, answer))

        # If not found in doc, fallback to general knowledge
        if not result.get("source_documents"):
            fallback = general_qa.run(query).strip()
            log_to_sheet(query, fallback)
            return fallback

        log_to_sheet(query, answer)
        return answer

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print("ERROR:", error_msg)
        return error_msg

#  Endpoints 
@app.post("/chat")
async def chat_endpoint(query: Query):
    response = smart_chat(query.question, query.user_id)
    return {"response": response}
@app.get("/ping")
async def ping():
    return {"status": "Up and running"}