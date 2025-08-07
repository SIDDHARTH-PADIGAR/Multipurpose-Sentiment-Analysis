import os
import torch
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# REQUEST/RESPONSE MODELS
# ========================================
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    text: str

# ========================================
# GLOBAL MODEL VARIABLES
# ========================================
model = None
tokenizer = None
device = None
label_names = ["Negative", "Positive"]

# ========================================
# MODEL LOADING FUNCTION
# ========================================
def load_model(model_path: str = "./final_model"):
    """Load the trained model and tokenizer into memory"""
    global model, tokenizer, device
    
    try:
        logger.info(f"Loading model from {model_path}...")
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device and move model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

# ========================================
# PREDICTION FUNCTION
# ========================================
def predict_sentiment(text: str) -> Dict[str, Any]:
    """Make sentiment prediction for given text"""
    try:
        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            "sentiment": label_names[predicted_class],
            "confidence": round(confidence, 4),
            "text": text
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ========================================
# FASTAPI APP SETUP
# ========================================
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using HuggingFace Transformers",
    version="1.0.0"
)

# ========================================
# API ENDPOINTS
# ========================================
@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    logger.info("Starting up server...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup!")
        raise RuntimeError("Model loading failed")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Sentiment Analysis API is running!",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    
    # Check if model is loaded
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate input
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400, 
            detail="Text field is required and cannot be empty"
        )
    
    # Check text length (prevent extremely long inputs)
    if len(request.text) > 5000:
        raise HTTPException(
            status_code=400, 
            detail="Text too long. Maximum 5000 characters allowed."
        )
    
    # Make prediction
    try:
        result = predict_sentiment(request.text.strip())
        return PredictionResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during prediction"
        )

# ========================================
# SERVER STARTUP
# ========================================
if __name__ == "__main__":
    logger.info("Starting Sentiment Analysis API server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )