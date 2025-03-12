from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Load Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Schema input untuk API
class NarasiRequest(BaseModel):
    text: str

# class QueryRequest(BaseModel):
#     prompt: str

# chat_history = []

# @app.post("/chat")
# def chat(query: QueryRequest):
#     response = llm.predict_messages([HumanMessage(content=query.prompt)])
#     chat_entry = {"input": query.prompt, "response": response.content}
#     chat_history.append(chat_entry)
#     return chat_entry
    
@app.post("/resume/")
async def resume_narasi(request: NarasiRequest):
    try:
        response = llm.predict_messages([
            HumanMessage(content=(
                f"Ringkas review aplikasi SIMPATIKA berikut dalam bentuk narasi yang menggambarkan pengalaman pengguna dengan mempertimbangkan sentimen dan aspek yang terkandung:\n\n"
                f"Komentar: \"{request.text}\"\n\n"
                f"Buat ringkasan yang jelas, agak panjang dan informatif serta saran selanjutnya untuk perbaikan dan pengembangan."
            ))
        ])
        return {"summary": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# def root():
#     return chat_history
