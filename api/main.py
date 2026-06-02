import os
import sys
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. Base Directory Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Load .env file from the parent directory (finalproj)
parent_dir = os.path.dirname(base_dir)
dotenv_path = os.path.join(parent_dir, ".env")
load_dotenv(dotenv_path)

# Import ML modules
from schemas import PredictionRequest, PredictionResponse
from predictor import CyberbullyingPredictor

app = FastAPI(
    title="Cyberbullying Detection API",
    description="A FastAPI server running a bidirectional LSTM model to classify cyberbullying."
)

# Supabase Client Initialization
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("Warning: Missing Supabase credentials. Dashboard will fail to load.")
    supabase = None

# Template configuration
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# 3. Bulletproof Paths: Dynamically generate absolute paths
MODEL_PATH = os.path.join(base_dir, "models", "optimized_nostop_bidirectional_lstm_model_10epoch.onnx")
TOKENIZER_PATH = os.path.join(base_dir, "models", "word_index.json")

# 4. Serverless Initialization
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

@app.get("/dashboard", response_class=HTMLResponse)
async def moderation_dashboard(request: Request):
    """Fetches flagged messages and renders the dashboard."""
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized.")
        
    response = supabase.table("warnings").select("*").order("timestamp", desc=True).execute()
    warnings = response.data
    
    # FIX: Use explicit keyword arguments for newer FastAPI/Starlette versions
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"warnings": warnings}
    )

@app.post("/dashboard/resolve/{warning_id}")
async def resolve_warning(warning_id: int):
    """Deletes a false-positive warning from Supabase."""
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized.")
        
    supabase.table("warnings").delete().eq("id", warning_id).execute()
    return RedirectResponse(url="/dashboard", status_code=303)