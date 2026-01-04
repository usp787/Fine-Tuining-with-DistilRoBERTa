#!/usr/bin/env python3
"""
FastAPI Server for LoRA Emotion Classifier
Exposes HTTP endpoints for multi-label emotion classification
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
import uvicorn
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GoEmotions 28 simplified labels
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# ============================================================================
# Model Architecture
# ============================================================================

class EmotionClassifier(nn.Module):
    """DistilRoBERTa with LoRA adapters + classification head"""
    
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(pooled)
        return logits


def build_model(model_path: str, rank: int = 16, device: str = 'cuda') -> EmotionClassifier:
    """Reconstruct and load the LoRA model"""
    logger.info(f"Loading model from: {model_path}")
    
    # 1. Load base backbone
    backbone = AutoModel.from_pretrained('distilroberta-base')
    
    # 2. Configure LoRA (must match training setup)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['key', 'query', 'value']
    )
    
    # 3. Inject LoRA adapters
    backbone = get_peft_model(backbone, peft_config)
    
    # 4. Build classifier head
    hidden_size = backbone.config.hidden_size
    classifier_head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, 28)
    )
    
    # 5. Combine into full model
    model = EmotionClassifier(backbone, classifier_head)
    
    # 6. Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded successfully on {device}")
    return model


# ============================================================================
# API Models (Request/Response schemas)
# ============================================================================

class PredictionRequest(BaseModel):
    """Request body for /predict endpoint"""
    text: str = Field(..., description="Text to classify", example="I'm so happy today!")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Probability threshold")
    top_k: Optional[int] = Field(None, ge=1, le=28, description="Return top K emotions")

class BatchPredictionRequest(BaseModel):
    """Request body for /predict/batch endpoint"""
    texts: List[str] = Field(..., description="List of texts to classify")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Probability threshold")
    top_k: Optional[int] = Field(None, ge=1, le=28, description="Return top K emotions")

class EmotionScore(BaseModel):
    """Single emotion with probability"""
    emotion: str
    probability: float

class PredictionResponse(BaseModel):
    """Response for single prediction"""
    text: str
    predicted_emotions: List[EmotionScore]
    all_scores: Optional[List[EmotionScore]] = None

class BatchPredictionResponse(BaseModel):
    """Response for batch prediction"""
    predictions: List[PredictionResponse]
    count: int


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="LoRA Emotion Classifier API",
    description="Multi-label emotion classification using DistilRoBERTa with LoRA adapters",
    version="1.0.0"
)

# Global variables (loaded at startup)
model = None
tokenizer = None
device = None

@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer when server starts"""
    global model, tokenizer, device
    
    # Configuration from environment variables
    model_path = os.getenv('MODEL_PATH', 'best_LoRA_model.pt')
    rank = int(os.getenv('LORA_RANK', '16'))
    device_name = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 60)
    logger.info("Starting LoRA Emotion Classifier API")
    logger.info("=" * 60)
    logger.info(f"Device: {device_name}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"LoRA rank: {rank}")
    
    # Load model
    try:
        device = torch.device(device_name)
        model = build_model(model_path, rank, device_name)
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        logger.info("✓ Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def predict_emotions(
    texts: List[str],
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    max_length: int = 128
) -> List[Dict]:
    """Run inference on input texts"""
    
    # Tokenize
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Format results
    results = []
    for i, text in enumerate(texts):
        pred_probs = probs[i]
        
        # Create emotion-probability pairs
        emotion_scores = [
            {'emotion': EMOTION_LABELS[j], 'probability': float(pred_probs[j])}
            for j in range(len(EMOTION_LABELS))
        ]
        
        # Sort by probability
        emotion_scores.sort(key=lambda x: x['probability'], reverse=True)
        
        # Filter predictions
        if top_k:
            predicted_emotions = emotion_scores[:top_k]
        else:
            predicted_emotions = [
                e for e in emotion_scores if e['probability'] >= threshold
            ]
        
        results.append({
            'text': text,
            'predicted_emotions': predicted_emotions,
            'all_scores': emotion_scores
        })
    
    return results


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "service": "LoRA Emotion Classifier API",
        "status": "running",
        "model_loaded": model is not None,
        "device": str(device),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "emotions": "/emotions"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint (required for AWS)"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }

@app.get("/emotions")
async def get_emotions():
    """Get list of supported emotion labels"""
    return {
        "emotions": EMOTION_LABELS,
        "count": len(EMOTION_LABELS)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Predict emotions for a single text
    
    Example request:
    ```json
    {
        "text": "I'm so happy today!",
        "threshold": 0.5,
        "top_k": 3
    }
    ```
    """
    try:
        results = predict_emotions(
            texts=[request.text],
            threshold=request.threshold,
            top_k=request.top_k
        )
        return results[0]
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict emotions for multiple texts
    
    Example request:
    ```json
    {
        "texts": [
            "I'm so happy today!",
            "This is terrible news.",
            "I'm not sure what to think."
        ],
        "threshold": 0.5
    }
    ```
    """
    try:
        if len(request.texts) > 100:
            raise HTTPException(
                status_code=400, 
                detail="Batch size too large (max 100 texts)"
            )
        
        results = predict_emotions(
            texts=request.texts,
            threshold=request.threshold,
            top_k=request.top_k
        )
        
        return {
            "predictions": results,
            "count": len(results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    # Run with: python app.py
    # Or use: uvicorn app:app --host 0.0.0.0 --port 8080
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
