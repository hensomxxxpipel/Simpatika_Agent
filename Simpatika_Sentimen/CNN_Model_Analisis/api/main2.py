from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Load Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Tambahkan Memory untuk menyimpan percakapan sebelumnya
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Buat prompt template yang menyertakan sejarah percakapan
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Ringkas review aplikasi SIMPATIKA berikut dalam bentuk narasi yang menggambarkan pengalaman pengguna dengan mempertimbangkan sentimen dan aspek yang terkandung:\n\n"
              "Komentar: \"{text}\"\n\n"
              "Buat ringkasan yang jelas, agak panjang dan informatif serta saran selanjutnya untuk pengembangan.")
])

# Buat LLMChain dengan memory
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Schema input untuk API
class NarasiRequest(BaseModel):
    text: str

@app.post("/resume/")
async def resume_narasi(request: NarasiRequest):
    try:
        response = conversation.run(text=request.text)
        return {"summary": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
