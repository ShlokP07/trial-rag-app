# main.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
from datetime import datetime, UTC
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

class QdrantRAG:
    def __init__(self):
        """Initialize the RAG system with OpenAI client and Qdrant"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize Qdrant client
        # In Docker, we'll connect to the official Qdrant container via host/port.
        # Locally (without Docker), you can still point this to a local Qdrant instance.
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Create collection if it doesn't exist
        try:
            self.qdrant.get_collection('documents')
        except Exception:
            self.qdrant.create_collection(
                collection_name='documents',
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=models.Distance.COSINE
                )
            )

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for a piece of text"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def process_pdf(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    async def process_document(self, file: UploadFile) -> str:
        """Process document: extract text, split into chunks, and store in Qdrant"""
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Read and process PDF
            content = await file.read()
            text = self.process_pdf(content)
            print(f"Extracted {len(text)} characters from PDF")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            print(f"Split into {len(chunks)} chunks")
            
            # Process chunks and store in Qdrant
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                embedding = self.get_embedding(chunk)
                point_id = str(uuid.uuid4())
                
                # Store in Qdrant
                self.qdrant.upsert(
                    collection_name="documents",
                    points=[models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "doc_id": doc_id,
                            "chunk_index": i,
                            "text": chunk,
                            "timestamp": datetime.now(UTC).isoformat()
                        }
                    )]
                )
            
            return doc_id
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Find the most relevant chunks for a query using Qdrant"""
        query_embedding = self.get_embedding(query)
        
        search_results = self.qdrant.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        
        return [result.payload['text'] for result in search_results]

    def answer_question(self, question: str) -> Dict:
        """Answer a question using relevant chunks and GPT"""
        relevant_chunks = self.find_relevant_chunks(question)
        context = "\n\n".join(relevant_chunks)
        
        messages = [
            {"role": "system", "content": """You are a helpful assistant that provides accurate, well-structured answers based on the given context. 
             If the answer cannot be fully found in the context, be clear about what you can and cannot answer. 
             Synthesize information from multiple chunks when necessary, and maintain the technical accuracy of the source material."""},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return {
            "answer": response.choices[0].message.content,
            "context": relevant_chunks
        }

# Initialize RAG system
rag = QdrantRAG()

@app.post("/upload")
async def upload_document(file: UploadFile):
    """Upload and process a PDF document"""
    doc_id = await rag.process_document(file)
    return {
        "message": "Document processed successfully",
        "doc_id": doc_id
    }

@app.post("/query")
async def query_documents(query: Query):
    """Query the document database"""
    try:
        result = rag.answer_question(query.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (frontend)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)