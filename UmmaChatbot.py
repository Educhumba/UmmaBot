import os
import logging
from collections import defaultdict
from time import time
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import threading
import uuid
import html

#logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

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
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

#  Load .env 
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("Missing GROQ_API_KEY in environment")

#loading chromaDB 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="db_groq", embedding_function=embeddings)

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

You are UmmaBot, an insurance assistant for Umma Insurance.
- Welcome and introduce yourself only in the first message, DO NOT introduce youself every time in evry message.
- Avoid slogans or repeated phrases.
- Do not hallucinate. If unsure, say so.
- Use bullet points or numbers for steps.
- Keep responses under 100 words unless needed.
- If the user asks for a human agent, provide company contacts or the specific one asked for.
- Answer naturally — no need to mention "from the provided information" unless necessary.
- Always be positive about the company when asked about it
- Don't make up a link if it is not provided in documents.
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

#  Google sheets logger 
def init_google_sheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("UmmaBot_service_key.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("UmmaBot Logs").sheet1
        return sheet
    except Exception as e:
        logger.error("Google Sheets Init Error", exc_info=True)
        return None
    
sheet_instance = init_google_sheets()
def log_to_sheet(user_query: str, bot_response: str):
    try:
        if sheet_instance:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet_instance.append_row([timestamp, user_query, bot_response])
    except Exception as e:
        logger.warning("Logging Error", exc_info=True)
def log_async(user_query: str, bot_response: str):
    """Run Google Sheets logging in the background so bot replies instantly."""
    threading.Thread(target=log_to_sheet, args=(user_query, bot_response), daemon=True).start()

#  Session Tracking 
user_sessions = {}
SESSION_TIMEOUT = timedelta(minutes=30)

def cleanup_sessions():
    now = datetime.now()
    expired = [uid for uid, data in user_sessions.items()
               if now - data["last_seen"] > SESSION_TIMEOUT]
    for uid in expired:
        del user_sessions[uid]

def get_user_history(user_id):
    cleanup_sessions()
    if user_id not in user_sessions:
        user_sessions[user_id] = {"history": [], "last_seen": datetime.now()}
    user_sessions[user_id]["last_seen"] = datetime.now()
    return user_sessions[user_id]["history"]

#Rate limiting for users(30 requests every 5 min)
REQUEST_LIMIT = 30
TIME_WINDOW = 300 
user_requests = defaultdict(list)

def rate_limited(user_id):
    now = time()
    requests = [req for req in user_requests[user_id] if now - req < TIME_WINDOW]
    if not requests:
        user_requests.pop(user_id, None)
    user_requests[user_id] = requests
    if len(requests) >= REQUEST_LIMIT:
        return True
    user_requests[user_id].append(now)
    return False

#  Core chat logic 
def smart_chat(query: str, user_id: str):
    if not query.strip():
        return "Please enter a valid question."
    try:
        #prompt injection sanitization
        safe_query = html.escape(query)
        # Maintain context per user for conversation flow and call main qa chain
        history = get_user_history(user_id)
        result = qa_chain.invoke({"question": safe_query, "chat_history": history})
        answer = result.get("answer", "").strip()

        #final answer decision
        final_answer = answer
        if not result.get("source_documents"):
            final_answer = general_qa.run(safe_query).strip()

        #append final answer to session history
        history.append((query, final_answer))

        #log to google sheets
        log_async(query, final_answer)
        return final_answer

    except Exception as e:
        logger.error("Error in smart chat function", exc_info=True)
        return f"An error occurred: {str(e)}"

#  Endpoints 
@app.post("/chat")
async def chat_endpoint(query: Query):
    if rate_limited(query.user_id):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    response = smart_chat(query.question, query.user_id)
    if response.startswith("An error occurred:"):
        raise HTTPException(status_code=500, detail=response)
    return {"response": response}

@app.get("/ping")
async def ping():
    return {"status": "Up and running"}