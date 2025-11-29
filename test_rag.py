# main.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
from datetime import datetime, UTC
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import io

# Directory where we persist original PDFs so the frontend can display them
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

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

    def process_pdf(self, content: bytes) -> List[Dict]:
        """Extract text from PDF content and preserve page boundaries.
        
        Returns:
            List of dicts: { "page_number": int, "text": str }
        """
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages = []
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                pages.append({
                    "page_number": page_number,
                    "text": page_text,
                })
            return pages
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    def clear_collection(self):
        """Clear all documents from the Qdrant collection before uploading a new one"""
        try:
            # Delete the entire collection
            self.qdrant.delete_collection('documents')
            print("Cleared existing collection")
        except Exception as e:
            print(f"Note: Collection may not exist yet: {e}")
        
        # Recreate the collection
        self.qdrant.create_collection(
            collection_name='documents',
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embedding dimension
                distance=models.Distance.COSINE
            )
        )
        print("Created fresh collection")

    async def process_document(self, file: UploadFile) -> str:
        """Process document: extract text, split into chunks, and store in Qdrant"""
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        try:
            # Clear old embeddings before processing new document
            self.clear_collection()
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Read PDF bytes
            content = await file.read()

            # Persist original PDF on disk so it can be viewed later
            pdf_path = os.path.join(DOCUMENTS_DIR, f"{doc_id}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(content)

            # Extract text for RAG (with page info)
            pages = self.process_pdf(content)
            total_chars = sum(len(p["text"]) for p in pages)
            print(f"Extracted {total_chars} characters from PDF across {len(pages)} pages")
            
            # Split each page into chunks and store with page metadata
            global_chunk_index = 0
            for page in pages:
                page_number = page["page_number"]
                page_text = page["text"]
                
                if not page_text.strip():
                    continue
                
                page_chunks = self.text_splitter.split_text(page_text)
                print(f"Page {page_number}: split into {len(page_chunks)} chunks")
                
                for local_idx, chunk in enumerate(page_chunks):
                    embedding = self.get_embedding(chunk)
                    point_id = str(uuid.uuid4())
                    
                    # Store in Qdrant with page metadata
                    self.qdrant.upsert(
                        collection_name="documents",
                        points=[models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "doc_id": doc_id,
                                "page": page_number,
                                "chunk_index": global_chunk_index,
                                "page_chunk_index": local_idx,
                                "text": chunk,
                                "timestamp": datetime.now(UTC).isoformat()
                            }
                        )]
                    )
                    global_chunk_index += 1
            
            return doc_id
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find the most relevant chunks for a query using Qdrant.
        
        Returns a list of dicts:
        {
            "text": str,
            "doc_id": str | None,
            "page": int | None,
            "chunk_index": int | None,
            "score": float | None,
        }
        """
        query_embedding = self.get_embedding(query)
        
        search_results = self.qdrant.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        formatted_results = []
        for result in search_results:
            payload = result.payload or {}
            text = payload.get("text", "")
            if not text:
                continue
            
            formatted_results.append({
                "text": text,
                "doc_id": payload.get("doc_id"),
                "page": payload.get("page"),
                "chunk_index": payload.get("chunk_index"),
                "score": float(getattr(result, "score", 0.0)),
            })
        
        return formatted_results

    def answer_question(self, question: str) -> Dict:
        """Answer a question using relevant chunks and GPT"""
        relevant_chunks = self.find_relevant_chunks(question)
        context = "\n\n".join(chunk["text"] for chunk in relevant_chunks if chunk.get("text"))
        
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
            "context": relevant_chunks  # Return full objects with page/doc_id
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


@app.get("/documents/{doc_id}.pdf")
async def get_document(doc_id: str):
    """Serve the original uploaded PDF so the frontend can display it."""
    pdf_path = os.path.join(DOCUMENTS_DIR, f"{doc_id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(pdf_path, media_type="application/pdf")

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