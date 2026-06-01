import os
import sys
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# Ensure the parent directory is in python path to resolve api module imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Load .env file
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

from api.schemas import PredictionRequest, PredictionResponse
from api.predictor import CyberbullyingPredictor

app = FastAPI(
    title="Cyberbullying Detection API",
    description="A FastAPI server running a bidirectional LSTM model to classify cyberbullying."
)

# MODEL_PATH = os.getenv("MODEL_PATH", "optimized_nostop_bidirectional_lstm_model.keras")
# MODEL_PATH = os.getenv("MODEL_PATH", "optimized_nostop_bidirectional_lstm_model_10epoch.keras")
# MODEL_PATH = os.getenv("MODEL_PATH", "optimized_nostop_bidirectional_lstm_model_10epoch.tflite")
MODEL_PATH = os.getenv("MODEL_PATH", "optimized_nostop_bidirectional_lstm_model_10epoch.onnx")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "word_index.json")

# Initialize global predictor
predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = CyberbullyingPredictor(MODEL_PATH, TOKENIZER_PATH)
        print("Model and Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Failed to load model/tokenizer during startup: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": predictor is not None}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    try:
        predictions = predictor.predict(request.text)
        
        # Flag as cyberbullying if any toxic category exceeds 50% probability
        THRESHOLD = 0.5
        detected = [label for label, prob in predictions.items() if prob >= THRESHOLD]
        is_cyberbullying = len(detected) > 0
        
        return PredictionResponse(
            text=request.text,
            predictions=predictions,
            is_cyberbullying=is_cyberbullying,
            detected_categories=detected
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
