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
        # In Docker, we'll connect to the official Qdrant container via host/port
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

    def _format_context_with_metadata(self, chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks with page numbers and metadata for better reference"""
        formatted_parts = []
        for idx, chunk in enumerate(chunks, 1):
            page = chunk.get("page", "?")
            text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            formatted_parts.append(
                f"[Context {idx} - Page {page} (Relevance: {score:.2f})]\n{text}"
            )
        return "\n\n---\n\n".join(formatted_parts)
    
    def _extract_answer_only(self, full_response: str) -> str:
        """Extract only the answer section from COT response, removing reasoning steps"""
        # Try to find the "Answer:" section
        answer_markers = [
            "**Answer:**",
            "**Answer:**\n",
            "Answer:",
            "Answer:\n",
            "## Answer",
            "## Answer\n"
        ]
        
        full_response_lower = full_response.lower()
        
        # Try each marker
        for marker in answer_markers:
            marker_lower = marker.lower()
            if marker_lower in full_response_lower:
                # Find the position (case-insensitive)
                idx = full_response_lower.find(marker_lower)
                if idx != -1:
                    # Get the actual marker with original case
                    actual_marker = full_response[idx:idx+len(marker)]
                    # Extract everything after the marker
                    answer_start = idx + len(actual_marker)
                    answer_text = full_response[answer_start:].strip()
                    
                    # Remove any trailing "Reasoning:" sections if they appear after
                    if "**reasoning:**" in answer_text.lower():
                        reasoning_idx = answer_text.lower().find("**reasoning:**")
                        answer_text = answer_text[:reasoning_idx].strip()
                    
                    if answer_text:
                        return answer_text
        
        # If no answer marker found, try to find content after "Reasoning:"
        reasoning_markers = ["**Reasoning:**", "**Reasoning:**\n", "Reasoning:", "Reasoning:\n"]
        for marker in reasoning_markers:
            marker_lower = marker.lower()
            if marker_lower in full_response_lower:
                idx = full_response_lower.find(marker_lower)
                if idx != -1:
                    # Look for answer after reasoning
                    after_reasoning = full_response[idx + len(marker):]
                    # Try to find answer section in the remaining text
                    for answer_marker in answer_markers:
                        answer_marker_lower = answer_marker.lower()
                        if answer_marker_lower in after_reasoning.lower():
                            answer_idx = after_reasoning.lower().find(answer_marker_lower)
                            actual_marker = after_reasoning[answer_idx:answer_idx+len(answer_marker)]
                            answer_start = answer_idx + len(actual_marker)
                            answer_text = after_reasoning[answer_start:].strip()
                            if answer_text:
                                return answer_text
        
        # Fallback: if no structured format found, return the full response
        # (might be a direct answer without reasoning section)
        return full_response.strip()
    
    def answer_question(self, question: str) -> Dict:
        """Answer a question using relevant chunks and GPT with Chain of Thought prompting"""
        relevant_chunks = self.find_relevant_chunks(question)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find any relevant information in the document to answer this question. Please try rephrasing your question or ensure a document has been uploaded.",
                "context": []
            }
        
        # Format context with page numbers and metadata
        formatted_context = self._format_context_with_metadata(relevant_chunks)
        
        # Enhanced Chain of Thought system prompt
        system_prompt = """You are an expert document analysis assistant. Your task is to answer questions based on the provided document context using Chain of Thought reasoning.

Follow this step-by-step process:

1. **Understand the Question**: Carefully read and understand what is being asked. Identify key concepts, entities, and the type of answer expected.

2. **Analyze the Context**: 
   - Review each context chunk provided (they are numbered and include page references)
   - Identify which chunks are most relevant to the question
   - Note any relationships or connections between different chunks
   - Pay attention to page numbers for source tracking

3. **Extract Relevant Information**:
   - Extract specific facts, data, or statements from the context that relate to the question
   - If information appears in multiple chunks, synthesize it coherently
   - Distinguish between main points and supporting details

4. **Reason Through the Answer**:
   - Think step by step about how the extracted information answers the question
   - If the question requires inference, explain your reasoning process
   - If information is incomplete, identify what can and cannot be answered

5. **Formulate the Answer**:
   - Provide a clear, well-structured answer
   - Use the exact terminology and phrasing from the document when possible
   - Include page references (e.g., "According to page X...") when citing specific information
   - If synthesizing from multiple pages, mention the relevant pages

6. **Verify Accuracy**:
   - Ensure your answer is directly supported by the provided context
   - Do not add information not present in the context
   - If the context doesn't fully answer the question, clearly state the limitations

**Important Guidelines**:
- Base your answer ONLY on the provided context chunks
- If the answer cannot be found in the context, explicitly state this
- Maintain technical accuracy and preserve the original meaning
- Use clear, professional language
- Structure longer answers with paragraphs or bullet points when appropriate"""

        # User prompt with Chain of Thought structure
        user_prompt = f"""Please answer the following question using Chain of Thought reasoning.

**Document Context:**
{formatted_context}

**Question:** {question}

**Instructions:**
Think through your answer step by step:
1. First, identify which context chunks are most relevant
2. Extract the key information needed to answer the question
3. Reason through how this information answers the question
4. Formulate your final answer with proper citations to page numbers

Provide your answer in the following format:

**Reasoning:**
[Your step-by-step analysis here]

**Answer:**
[Your final, well-structured answer here with page references when applicable]"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused, consistent reasoning
            max_tokens=1000  # Increased to accommodate reasoning steps
        )
        
        # Extract only the answer section, removing reasoning steps
        full_response = response.choices[0].message.content
        clean_answer = self._extract_answer_only(full_response)
        
        return {
            "answer": clean_answer,
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