import os
import logging
from collections import defaultdict
from time import time
import time as time_module
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from threading import Lock, Thread
import uuid
import html
import shutil

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

#  Request Models
class Query(BaseModel):
    question: str
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class FeedbackData(BaseModel):
    user_id: str
    rating: str
    feedback: str = ""
    final: bool = False #final submission or skip

#  Load .env 
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("Missing GROQ_API_KEY in environment")

service_key_path = "UmmaBot_service_key.json"
if not os.path.exists(service_key_path):
    raise FileNotFoundError(f"Missing google service key file: {service_key_path}")

#loading chromaDB 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = Chroma(persist_directory="db_groq", embedding_function=embeddings)

#LLM Setup 
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-8b-8192",
    temperature=0.3,
    max_tokens=512
)

#Prompt Template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""

You are UmmaBot, an insurance assistant for Umma Insurance.
- Welcome and introduce yourself only in the first message. Do NOT introduce yourself in every reply.
- Avoid slogans or repeated phrases.
- Do not hallucinate. If unsure, say so.
- Use bullet points or numbers for steps.
- Keep responses under 100 words unless needed.
- If the user asks for a human agent, provide company contacts or the specific one asked for.
- Answer naturally — no need to mention "from the provided information" unless necessary.
- Only answer based on the current user question. Do not repeat answers to previous questions unless explicitly asked to do so, or is a context continuation
- Always be positive about the company when asked about it
- Don't make up a link if it is not provided in documents.
{context}

User Question: {question}
Your Response:"""
)

def load_vectordb():
    try:
        with open("active_db.txt", "r") as f:
            persist_dir = f.read().strip()
    except FileNotFoundError:
        persist_dir = "db_groq" 
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Initial load
vectordb = load_vectordb()

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

@app.post("/admin/reload_db")
async def reload_db():
    global vectordb, qa_chain, general_qa
    try:
        vectordb = load_vectordb()
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
        return {"status": "success", "message": "Vector DB reloaded from active_db.txt"}
    except Exception as e:
        logger.error("DB reload failed", exc_info=True)
        return {"status": "error", "message": str(e)}

#  Google sheets logger 
sheet_instance = None
sheet_lock = Lock()
def init_google_sheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(service_key_path, scope)
        client = gspread.authorize(creds)
        sheet = client.open("UmmaBot Logs").sheet1
        return sheet
    except Exception:
        logger.error("Google Sheets Initialization failed", exc_info=True)
        return None
    
sheet_instance = init_google_sheets()

#Append to google sheets with thread seafty
def append_to_sheet(row):
    global sheet_instance
    with sheet_lock:
        try:
            if not sheet_instance:
                sheet_instance = init_google_sheets()
            if sheet_instance:
                sheet_instance.append_row(row)
        except Exception:
            logger.error("Failed to append to google sheets", exc_info=True)
    
#log full conversation to google sheets
def log_full_conversation(user_id, rating="", feedback=""):
    try:
        session = user_sessions.get(user_id)
        if not session or session.get("logged"):
            return
        if rating or feedback:
            session["last_rating"] = rating
            session["last_feedback"] = feedback

        rating = rating or session.get("last_rating", "")
        feedback = feedback or session.get("last_feedback", "")

        # Prepare conversation transcript
        transcript = "\n".join([f"User: {q}\nBot: {a}" for q,a in session["history"]])
        today = datetime.now().strftime("%Y-%m-%d") 
        row = [
            today,
            session.get("conversation_id", ""),
            user_id,
            session.get("start_time", "").strftime("%Y-%m-%d %H:%M:%S"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            transcript.strip(),
            rating ,
            feedback 
        ]
        Thread(target=append_to_sheet, args=(row,), daemon=True).start()
        session["logged"] = True
    except Exception:
        logger.error("Error logging full conversation", exc_info=True)

#  Session Tracking 
user_sessions = {}
SESSION_TIMEOUT = timedelta(minutes=30)

def cleanup_sessions():
    now = datetime.now()
    expired = []
    for uid, data in list(user_sessions.items()):
        if now - data["last_seen"] > SESSION_TIMEOUT:
            if not data.get("end_time"):
                data["end_time"] = now
            expired.append(uid)
            if not data.get("logged"):
                log_full_conversation(uid)
    for uid in expired:
        user_sessions.pop(uid, None)

def get_user_history(user_id):
    cleanup_sessions()
    now = datetime.now()
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "conversation_id": f"{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:5]}",
            "history": [],
            "start_time": now,
            "last_seen": now,
            "end_time": None,
            "logged": False,
            "feedback_given": False
        }
    user_sessions[user_id]["last_seen"] = now
    return user_sessions[user_id]["history"]

#Rate limiting for users(30 requests every 5 min)
REQUEST_LIMIT = 30
TIME_WINDOW = 300 
user_requests = defaultdict(list)

def rate_limited(user_id):
    now = time()
    requests = [req for req in user_requests[user_id] if now - req < TIME_WINDOW]
    user_requests[user_id] = requests
    if len(requests) >= REQUEST_LIMIT:
        return True
    user_requests[user_id].append(now)
    return False

#  Core chat logic 
def smart_chat(query: str, user_id: str):
    if not query.strip():
        return "Please enter a valid question."
    # Detect explicit end-of-chat phrases
    end_phrases = ["thanks", "thank you", "bye", "goodbye", "that's all", "no, i'm done", "no thanks", "cool", "adios"]
    if any(phrase in query.lower() for phrase in end_phrases):
        # Mark session as ended
        session = user_sessions.get(user_id)
        if session:
            session["end_time"] = datetime.now()
        return "__END_CONVERSATION__"
    try:
        #prompt injection sanitization
        safe_query = html.escape(query)
        # Maintain context per user for conversation flow and call main qa chain
        history = get_user_history(user_id)
        try:
            result = qa_chain.invoke({"question": safe_query, "chat_history": history})
            answer = result.get("answer", "").strip()
        except Exception:
            logger.error("Error in QA chain", exc_info=True)
            answer = ""
        if not answer:
            try:
                answer = general_qa.run(safe_query).strip()
            except Exception:
                logger.error("Error in general QA fallback", exc_info=True)
                answer = "I am sorry i couldn't find the answer to you question"
        #append final answer to session history
        history.append((query, answer))
        return answer

    except Exception:
        logger.error("Error in smart chat function", exc_info=True)
        return "An error occurred: Unable to process request."

#  Endpoints 
@app.post("/chat")
async def chat_endpoint(query: Query):
    if rate_limited(query.user_id):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    
    # Check if session is already ended by inactivity
    session = user_sessions.get(query.user_id)
    if session and not session.get("feedback_given", False) and not session.get("feedback_shown", False):
        if session.get("end_time") or (datetime.now() - session["last_seen"]>SESSION_TIMEOUT):   
            session["feedback_shown"] = True
            return {"response": "__SHOW_FEEDBACK__"}

    response = smart_chat(query.question, query.user_id)
    if response == "__END_CONVERSATION__":
        return {"response": "__SHOW_FEEDBACK__"}
    
    if response.startswith("An error occurred:"):
        raise HTTPException(status_code=500, detail=response)
    return {"response": response}

@app.post("/feedback")
async def feedback_endpoint(data: FeedbackData):
    try:
        session = user_sessions.get(data.user_id)
        if not session:
            return {"status": "error", "message": "Session not found"}

        # If this is just a rating selection (not final), store and return
        if not data.final and data.rating:
            session["last_rating"] = data.rating
            return {"status": "ok", "message": "Rating stored"}

        # If final submission (submit feedback or skip)
        # Use passed rating if present, otherwise use stored rating
        rating = data.rating or session.get("last_rating", "")
        feedback = data.feedback or session.get("last_feedback", "")

        # store last_rating/last_feedback for consistency
        if rating:
            session["last_rating"] = rating
        if feedback:
            session["last_feedback"] = feedback
        #Mark feedback complete
        session["feedback_given"] = True

        # log and remove session
        log_full_conversation(data.user_id, rating, feedback)
        user_sessions.pop(data.user_id, None)
        return {"status": "ok", "message": "Feedback logged"}
    except Exception:
        logger.error("Error in feedback endpoint", exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to save feedback")

#Background cleanup
def background_cleanup():
    while True:
        cleanup_sessions()
        time_module.sleep(60)

Thread(target=background_cleanup, daemon=True).start()
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectordb, qa_chain, general_qa
    try:
        if not file.filename.endswith(".pdf"):
            return {"status": "error", "message": "Only PDF files are allowed."}
        # save uploaded PDF temporarily
        temp_path = f"uploaded_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # load and split
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        # create new unique persist dir
        persist_dir = f"db_groq/{uuid.uuid4()}"
        os.makedirs(persist_dir, exist_ok=True)
        # build new DB
        vectordb = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=persist_dir
        )
        # write active path
        with open("active_db.txt", "w") as f:
            f.write(persist_dir)
        # re-init QA chains with new DB
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
        # cleanup temp file
        os.remove(temp_path)
        return {"status": "success", "message": f"New training document processed. Active DB: {persist_dir}"}

    except Exception as e:
        logger.error("Upload processing failed", exc_info=True)
        return {"status": "error", "message": str(e)}

    except Exception as e:
        logger.error("Upload processing failed", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.get("/ping")
async def ping():
    return {"status": "Up and running"}