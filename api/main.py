import os
import sys
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# 1. Fix Directory Traversal: Only go up ONE level because 'api' is the root
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Load .env file (Works locally, safely ignored on Vercel)
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

# 2. Fix Imports: Remove the 'api.' prefix since we are already inside the api folder
from schemas import PredictionRequest, PredictionResponse
from predictor import CyberbullyingPredictor

app = FastAPI(
    title="Cyberbullying Detection API",
    description="A FastAPI server running a bidirectional LSTM model to classify cyberbullying."
)

# 3. Fix Paths: Remove 'api/' prefix and dynamically map them to the models subfolder
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(base_dir, "models", "optimized_nostop_bidirectional_lstm_model_10epoch.onnx"))
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", os.path.join(base_dir, "models", "word_index.json"))

# 4. Serverless Initialization: Load globally instead of using @app.on_event("startup")
try:
    predictor = CyberbullyingPredictor(MODEL_PATH, TOKENIZER_PATH)
    print("Model and Tokenizer loaded successfully!")
except Exception as e:
    print(f"Failed to load model/tokenizer during cold start: {e}")
    predictor = None

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": predictor is not None}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet or failed to initialize.")
    
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