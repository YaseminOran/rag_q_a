from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, NotebookLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import uvicorn
import tempfile
from pathlib import Path
import json

# .env dosyasını yükle
load_dotenv()

app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API for Multiple File Types Q&A",
    version="1.0.0"
)

# OpenAI API anahtarını .env dosyasından al
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")

# Global değişkenler
qa_chain = None
vectorstore = None
llm = None  # LLM'i global olarak tanımla

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    question: str
    answer: str

class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str

class Quiz(BaseModel):
    title: str
    questions: list[QuizQuestion]

class QuizRequest(BaseModel):
    history_file: str

def load_document(file_path: str):
    """Dosya türüne göre uygun loader'ı seç ve yükle"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return PyPDFLoader(file_path).load()
    elif file_extension == '.ipynb':
        return NotebookLoader(file_path).load()
    elif file_extension == '.txt':
        return TextLoader(file_path).load()
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {file_extension}")

def load_all_files(directory: str):
    """Belirtilen klasördeki tüm desteklenen dosyaları yükle"""
    all_documents = []
    directory_path = Path(directory)
    
    # Desteklenen dosya uzantıları
    supported_extensions = {'.pdf', '.ipynb', '.txt'}
    
    # Klasördeki tüm desteklenen dosyaları bul
    files = []
    for ext in supported_extensions:
        files.extend(directory_path.glob(f'**/*{ext}'))
    
    if not files:
        raise ValueError(f"'{directory}' klasöründe desteklenen dosya bulunamadı! (Desteklenen: {', '.join(supported_extensions)})")
    
    print(f"\nBulunan dosyalar:")
    for file in files:
        print(f"- {file.name}")
        try:
            # Her dosyayı yükle
            documents = load_document(str(file))
            all_documents.extend(documents)
            print(f"  ✓ Başarıyla yüklendi")
        except Exception as e:
            print(f"  ✗ Yüklenemedi: {str(e)}")
    
    return all_documents

def initialize_rag(directory: str = None):
    global qa_chain, vectorstore, llm
    
    try:
        if directory:
            # Klasörden tüm dosyaları yükle
            documents = load_all_files(directory)
        else:
            # Varsayılan olarak dokumanlar klasörünü kullan
            default_dir = "/Users/yaseminarslan/Desktop/mcp/dokumanlar"
            if os.path.exists(default_dir):
                documents = load_all_files(default_dir)
            else:
                print("\nDokumanlar klasörü bulunamadı. Sistem boş başlatılıyor...")
                documents = []

        # Eğer hiç doküman yoksa, boş bir sistem başlat
        if not documents:
            print("\nHiç doküman yüklenmedi. Sistem boş başlatılıyor...")
            vectorstore = None
            qa_chain = None
            llm = ChatOpenAI(temperature=0)
            return

        # Chunk işlemi
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Embedding + FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # LLM'i oluştur
        llm = ChatOpenAI(temperature=0)
        
        # Retrieval + QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        
        print("\nRAG sistemi başarıyla başlatıldı!")
        
    except Exception as e:
        print(f"\nHata: {str(e)}")
        # Hata durumunda boş bir sistem başlat
        vectorstore = None
        qa_chain = None
        llm = ChatOpenAI(temperature=0)
        print("Sistem boş başlatıldı.")

@app.on_event("startup")
async def startup_event():
    print("RAG sistemini başlatıyorum...")
    initialize_rag()
    print("RAG sistemi hazır!")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Yeni bir dosya yükle ve RAG sistemini güncelle"""
    try:
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # RAG sistemini güncelle
            initialize_rag(temp_file.name)
            
            # Geçici dosyayı sil
            os.unlink(temp_file.name)
            
        return {"message": f"{file.filename} başarıyla yüklendi ve RAG sistemi güncellendi"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-files")
async def load_files(directory: str):
    """Belirtilen klasördeki tüm desteklenen dosyaları yükle"""
    try:
        initialize_rag(directory)
        return {"message": f"'{directory}' klasöründeki dosyalar başarıyla yüklendi"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(question: Question):
    try:
        # Bağlamı kontrol et
        context = ""
        if "Önceki konuşma bağlamı:" in question.question:
            parts = question.question.split("\n\nYeni soru: ")
            if len(parts) == 2:
                context = parts[0]
                question.question = parts[1]
        
        # Önce yüklenen dosyalarda ara
        if vectorstore is not None:
            docs = vectorstore.similarity_search(question.question, k=3)
            if docs:
                # Bağlamı prompt'a ekle
                prompt = f"""Aşağıdaki bağlamı ve soruyu dikkate alarak cevap ver.
                
                {context}
                
                Soru: {question.question}
                
                Eğer soru önceki bağlamla ilgiliyse, cevabını bağlamı dikkate alarak oluştur.
                Eğer soru farklı bir konuyla ilgiliyse, bağlamı göz ardı edebilirsiniz.
                Cevabını "[Yüklenen Dosyalardan] " öneki ile başlat.
                """
                
                answer = qa_chain.run(prompt)
                return {"answer": answer}
        
        # Eğer yüklenen dosyalarda cevap bulunamazsa, OpenAI'ya sor
        prompt = f"""Aşağıdaki bağlamı ve soruyu dikkate alarak cevap ver.
        
        {context}
        
        Soru: {question.question}
        
        Eğer soru önceki bağlamla ilgiliyse, cevabını bağlamı dikkate alarak oluştur.
        Eğer soru farklı bir konuyla ilgiliyse, bağlamı göz ardı edebilirsiniz.
        
        Eğer soru makine öğrenmesi, yapay zeka, veri bilimi, programlama veya ilgili teknik konularla ilgiliyse, detaylı ve teknik bir açıklama yap.
        Cevabını "[OpenAI Genel Bilgi Tabanından] " öneki ile başlat.
        """
        
        # OpenAI'ya sor
        response = llm.predict(prompt)
        return {"answer": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-quiz")
async def generate_quiz(request: QuizRequest):
    try:
        # Geçmiş soruları ve cevapları al
        if not os.path.exists(request.history_file):
            raise HTTPException(status_code=404, detail="Oturum geçmişi bulunamadı!")
            
        with open(request.history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
            
        if not history:
            raise HTTPException(status_code=404, detail="Oturum geçmişi boş!")
        
        # Quiz oluşturma prompt'u
        prompt = f"""
        Aşağıdaki soru-cevap geçmişine dayanarak 5 soruluk bir quiz oluştur:
        
        {json.dumps(history, ensure_ascii=False, indent=2)}
        
        Her soru için:
        1. Soru metni
        2. 4 şık
        3. Doğru cevap
        
        Format:
        {{
            "title": "Konu Başlığı",
            "questions": [
                {{
                    "question": "Soru metni",
                    "options": ["A) Şık 1", "B) Şık 2", "C) Şık 3", "D) Şık 4"],
                    "correct_answer": "Doğru şık"
                }}
            ]
        }}
        """
        
        # Quiz oluştur
        response = llm.predict(prompt)
        
        # JSON formatına çevir
        quiz_data = json.loads(response)
        return Quiz(**quiz_data)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Quiz oluşturulurken bir hata oluştu: JSON formatı geçersiz")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz oluşturulurken bir hata oluştu: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_initialized": qa_chain is not None
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5001) 