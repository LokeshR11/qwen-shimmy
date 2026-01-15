"""
Simple embedding server using sentence-transformers
Reliable, production-ready, and works with SAP AI Core
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union
import uvicorn
from sentence_transformers import SentenceTransformer
import numpy as np

# Configuration
PORT = int(os.getenv("PORT", 8080))
HOST = "0.0.0.0"

# Model will be downloaded automatically on first run
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize FastAPI
app = FastAPI(
    title="Embedding API Server",
    description="Production-ready embedding service",
    version="1.0.0"
)

model = None

# Request/Response models
class EmbeddingRequest(BaseModel):
    inputs: Union[str, List[str]]
    normalize: Optional[bool] = True

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int

class SimilarityRequest(BaseModel):
    text1: str
    text2: str

class SimilarityResponse(BaseModel):
    similarity: float

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    
    print("=" * 60)
    print("🚀 Embedding Server Starting")
    print("=" * 60)
    print(f"📦 Model: {MODEL_NAME}")
    print("📥 Loading model...")
    
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("✅ Model loaded successfully!")
        print(f"📊 Embedding dimension: 384")
        print("=" * 60)
        print("🎉 Server Ready!")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Embedding API Server",
        "model": MODEL_NAME,
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "embeddings": "POST /v1/embeddings",
            "similarity": "POST /v1/similarity",
            "health": "GET /health",
            "info": "GET /info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for SAP AI Core"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_NAME
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": MODEL_NAME,
        "embedding_dimension": 384,
        "max_sequence_length": 256,
        "framework": "sentence-transformers"
    }

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        texts = [request.inputs] if isinstance(request.inputs, str) else request.inputs
        
        print(f"📝 Creating embeddings for {len(texts)} text(s)")
        
        embeddings = model.encode(
            texts,
            normalize_embeddings=request.normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        print(f"✅ Generated embeddings with shape {embeddings.shape}")
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model=MODEL_NAME,
            dimension=embeddings.shape[1]
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """Compute cosine similarity between two texts"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        print(f"📝 Computing similarity")
        
        embeddings = model.encode(
            [request.text1, request.text2],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        
        print(f"✅ Similarity: {similarity:.4f}")
        
        return SimilarityResponse(similarity=similarity)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"🌐 Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
