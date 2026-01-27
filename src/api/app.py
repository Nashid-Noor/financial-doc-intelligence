"""
FastAPI Application
Financial Document Intelligence Platform

RESTful API for document upload, processing, and querying.
"""

import os
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_processing import PDFParser, FinancialChunker, Chunk
from retrieval import FinancialEmbedder, VectorStore, HybridRetriever
from reasoning import NumericalReasoner, CitationManager, Citation
from model import FinancialQAModel


# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for document queries."""
    question: str = Field(..., description="Question to answer")
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    company: Optional[str] = Field(None, description="Filter by company ticker")
    filing_type: Optional[str] = Field(None, description="Filter by filing type (10-K, 10-Q)")
    top_k: int = Field(10, ge=1, le=50, description="Number of chunks to retrieve")
    include_citations: bool = Field(True, description="Include source citations")
    include_reasoning: bool = Field(False, description="Include reasoning steps for calculations")


class CitationResponse(BaseModel):
    """Citation information in response."""
    company: str
    filing_type: str
    fiscal_year: Optional[int]
    page: Optional[int]
    section: Optional[str]
    text_snippet: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[CitationResponse]
    confidence: float
    reasoning_steps: Optional[List[str]] = None
    processing_time: float
    query_id: str


class DocumentInfo(BaseModel):
    """Information about an uploaded document."""
    document_id: str
    filename: str
    company: Optional[str]
    filing_type: Optional[str]
    fiscal_year: Optional[int]
    total_pages: int
    total_chunks: int
    uploaded_at: str
    status: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, str]


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.documents: Dict[str, DocumentInfo] = {}
        self.chunks: Dict[str, List[Chunk]] = {}  # document_id -> chunks
        
        # Initialize components (lazy loading)
        self._pdf_parser = None
        self._chunker = None
        self._embedder = None
        self._vector_store = None
        self._retriever = None
        self._qa_model = None
        self._numerical_reasoner = None
        self._citation_manager = None
    
    @property
    def pdf_parser(self) -> PDFParser:
        if self._pdf_parser is None:
            self._pdf_parser = PDFParser()
        return self._pdf_parser
    
    @property
    def chunker(self) -> FinancialChunker:
        if self._chunker is None:
            self._chunker = FinancialChunker(chunk_size=512, overlap=50)
        return self._chunker
    
    @property
    def embedder(self) -> FinancialEmbedder:
        if self._embedder is None:
            # Force MiniLM usage to align with config
            self._embedder = FinancialEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return self._embedder
    
    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(path=":memory:")
            self._vector_store.create_collection(
                vector_size=self.embedder.embedding_dim
            )
        return self._vector_store
    
    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedder=self.embedder
            )
        return self._retriever
    
    @property
    def qa_model(self) -> FinancialQAModel:
        if self._qa_model is None:
            self._qa_model = FinancialQAModel()
        return self._qa_model

    @property
    def numerical_reasoner(self) -> NumericalReasoner:
        if self._numerical_reasoner is None:
            self._numerical_reasoner = NumericalReasoner()
        return self._numerical_reasoner
    
    @property
    def citation_manager(self) -> CitationManager:
        if self._citation_manager is None:
            self._citation_manager = CitationManager()
        return self._citation_manager


# Global state
state = AppState()


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Financial Document Intelligence API",
        description="RAG-powered Q&A system for SEC filings and financial documents",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Financial Document Intelligence API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {
        "api": "healthy",
        "vector_store": "healthy" if state._vector_store else "not_initialized",
        "embedder": "healthy" if state._embedder else "not_initialized"
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components=components
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a financial document.
    
    Supported formats: PDF
    
    Processing steps:
    1. Parse PDF and extract text/tables
    2. Chunk document
    3. Generate embeddings
    4. Store in vector database
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Generate document ID
    doc_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{doc_id}_{file.filename}"
    
    try:
        # Save file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Parse PDF
        logger.info(f"Parsing document: {file.filename}")
        parsed_doc = state.pdf_parser.parse_filing(str(temp_path))
        
        # Chunk document
        logger.info("Chunking document")
        chunks = state.chunker.chunk_document(parsed_doc)
        
        # Add document ID to chunk metadata
        for chunk in chunks:
            chunk.metadata['document_id'] = doc_id
        
        # Generate embeddings
        logger.info("Generating embeddings")
        embeddings = state.embedder.encode_chunks(chunks)
        
        # Add to vector store
        logger.info("Adding to vector store")
        state.vector_store.add_chunks(chunks, embeddings)
        
        # Update BM25 index
        state.retriever.build_bm25_index(
            state.retriever.bm25_chunks + chunks
        )
        
        # Store document info
        doc_info = DocumentInfo(
            document_id=doc_id,
            filename=file.filename,
            company=parsed_doc.metadata.get('company_name'),
            filing_type=parsed_doc.metadata.get('filing_type'),
            fiscal_year=parsed_doc.metadata.get('fiscal_year'),
            total_pages=parsed_doc.total_pages,
            total_chunks=len(chunks),
            uploaded_at=datetime.now().isoformat(),
            status="processed"
        )
        state.documents[doc_id] = doc_info
        state.chunks[doc_id] = chunks
        
        processing_time = time.time() - start_time
        
        return UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            status="success",
            message=f"Document processed: {len(chunks)} chunks created",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Answer a question about uploaded documents.
    
    Processing steps:
    1. Retrieve relevant chunks using hybrid search
    2. Check if calculation is needed
    3. Generate answer (mock or real model)
    4. Extract citations
    5. Calculate confidence score
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())[:8]
    
    # Build filters
    filters = {}
    if request.document_ids:
        filters['document_id'] = request.document_ids
    if request.company:
        filters['company'] = request.company
    if request.filing_type:
        filters['filing_type'] = request.filing_type
    
    # Retrieve relevant chunks
    logger.info(f"Retrieving chunks for: {request.question}")
    results = state.retriever.retrieve(
        query=request.question,
        top_k=request.top_k,
        filters=filters if filters else None
    )
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Please upload documents first."
        )
    
    # Extract chunks and scores
    chunks = [r.chunk for r in results]
    retrieval_scores = [r.combined_score for r in results]
    
    # Check if calculation is needed
    reasoning_steps = None
    needs_calc, op_type = state.numerical_reasoner.requires_calculation(request.question)
    
    if needs_calc and request.include_reasoning:
        # Extract numbers and perform calculation
        context_text = "\n".join(c.text for c in chunks)
        numbers = state.numerical_reasoner.extract_numbers(context_text)
        
        if numbers and op_type:
            calc_result = state.numerical_reasoner.calculate(op_type, numbers)
            if calc_result:
                reasoning_steps = [
                    f"Operation: {calc_result.operation.value}",
                    f"Formula: {calc_result.formula}",
                    f"Result: {calc_result.formatted_answer}"
                ]
    
    # Generate answer (using real model)
    answer = state.qa_model.generate_answer(
        question=request.question,
        context_chunks=chunks,
        config=None  # Use default config
    )
    
    # Append reasoning steps to the answer if they exist
    if reasoning_steps:
        reasoning_text = "\n\nReasoning:\n" + "\n".join(f"- {step}" for step in reasoning_steps)
        answer += reasoning_text
    
    # Extract citations
    citations = []
    confidence = 0.0
    
    if request.include_citations:
        citation_objs = state.citation_manager.extract_citations(
            answer, chunks, retrieval_scores
        )
        confidence, _ = state.citation_manager.calculate_confidence(
            answer, chunks, retrieval_scores
        )
        
        citations = [
            CitationResponse(
                company=c.company,
                filing_type=c.filing_type,
                fiscal_year=c.fiscal_year,
                page=c.page,
                section=c.section,
                text_snippet=c.text_snippet,
                relevance_score=c.relevance_score
            )
            for c in citation_objs
        ]
    
    processing_time = time.time() - start_time
    
    return QueryResponse(
        answer=answer,
        sources=citations,
        confidence=confidence,
        reasoning_steps=reasoning_steps,
        processing_time=processing_time,
        query_id=query_id
    )


def _generate_mock_answer(question: str, chunks: List[Chunk], reasoning: List[str] = None) -> str:
    """Generate a mock answer based on retrieved chunks."""
    if not chunks:
        return "I don't have enough information to answer this question."
    
    # Extract key information from chunks
    context_parts = []
    for chunk in chunks[:3]:  # Use top 3 chunks
        context_parts.append(chunk.text[:200])
    
    # Build mock answer
    answer_parts = [
        f"Based on the financial documents provided, "
    ]
    
    # Add some context from chunks
    first_chunk = chunks[0]
    company = first_chunk.metadata.get('company', 'the company')
    filing_type = first_chunk.metadata.get('filing_type', 'filing')
    
    answer_parts.append(
        f"from {company}'s {filing_type}, "
    )
    
    # Include reasoning if available
    if reasoning:
        answer_parts.append(f"the calculation shows: {reasoning[-1]}. ")
    else:
        answer_parts.append(
            f"the relevant information indicates: {chunks[0].text[:150]}... "
        )
    
    answer_parts.append(
        f"(Source: {first_chunk.metadata.get('company', 'Unknown')} "
        f"{first_chunk.metadata.get('filing_type', '')}, "
        f"Page {first_chunk.metadata.get('page_number', 'N/A')})"
    )
    
    return "".join(answer_parts)


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents."""
    return list(state.documents.values())


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """Get details for a specific document."""
    if doc_id not in state.documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {doc_id}"
        )
    return state.documents[doc_id]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks."""
    if doc_id not in state.documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {doc_id}"
        )
    
    # Remove from vector store
    state.vector_store.delete_by_filter({'document_id': doc_id})
    
    # Remove from state
    del state.documents[doc_id]
    if doc_id in state.chunks:
        del state.chunks[doc_id]
    
    return {"status": "deleted", "document_id": doc_id}


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    vector_stats = state.vector_store.get_collection_stats()
    
    return {
        "total_documents": len(state.documents),
        "total_chunks": vector_stats.get('total_points', 0),
        "vector_store_status": vector_stats.get('status', 'unknown'),
        "companies": list(set(
            doc.company for doc in state.documents.values() 
            if doc.company
        ))
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Financial Document Intelligence API")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
