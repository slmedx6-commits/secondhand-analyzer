import os
import base64
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-startup model check/train trigger
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check if pricing model exists, if not, train it
    backend_dir = os.path.dirname(__file__) or '.'
    model_path = os.path.join(backend_dir, 'price_model.joblib')
    if not os.path.exists(model_path):
        logger.info("Pricing model 'price_model.joblib' not found. Training programmatically...")
        try:
            from .train_model import train_pricing_model
            train_pricing_model()
        except Exception as te:
            logger.error(f"Failed to auto-train local pricing model: {te}")
    else:
        logger.info(f"Local pricing model found at {model_path}.")
        
    # Pre-warm YOLOv8 and scikit-learn pricing models
    try:
        logger.info("Pre-warming YOLOv8 and Pricing models on startup...")
        from .analyzer import get_yolo_model, get_pricing_model
        get_yolo_model()
        get_pricing_model()
        logger.info("Models pre-warmed successfully!")
    except Exception as we:
        logger.error(f"Failed to pre-warm models: {we}")
        
    yield
    # Cleanup if needed

app = FastAPI(
    title="AI Second-Hand Item Analyzer",
    description="Decoupled web app utilizing computer vision and machine learning for estimating physical item value and condition.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema for base64 uploads (webcam capture)
class Base64UploadRequest(BaseModel):
    image_base64: str  # Data URL or raw base64 string

@app.get("/stats")
async def get_system_stats():
    """
    Get system monitoring stats (CPU & RAM) to present on frontend header.
    Shows the user that execution is resource-friendly.
    """
    try:
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_info = psutil.virtual_memory()
        
        # CPU Temperature (not available on all Windows platforms, fallback is None)
        return {
            "cpu_usage_percent": cpu_usage,
            "ram_usage_percent": ram_info.percent,
            "ram_available_mb": round(ram_info.available / (1024 * 1024), 1),
            "ram_total_mb": round(ram_info.total / (1024 * 1024), 1)
        }
    except Exception as e:
        logger.error(f"Error gathering system stats: {e}")
        return {"error": "Failed to read system stats."}

@app.post("/analyze")
async def analyze_file_upload(file: UploadFile = File(...)):
    """
    Receives an uploaded file, runs visual and price analysis, returns report.
    """
    try:
        # Validate MIME type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must be an image."
            )
            
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file uploaded."
            )
            
        from .analyzer import analyze_item
        report = analyze_item(image_bytes)
        
        if not report.get("success", False):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": report.get("error", "Unknown error during analysis.")}
            )
            
        return report
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /analyze file upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )

@app.post("/analyze-base64")
async def analyze_base64_upload(payload: Base64UploadRequest):
    """
    Receives base64 encoded image string (e.g. from client webcam), decodes and analyzes.
    """
    try:
        data = payload.image_base64
        # Remove prefix if standard HTML5 dataurl (e.g. "data:image/jpeg;base64,")
        if "," in data:
            data = data.split(",")[1]
            
        image_bytes = base64.b64decode(data)
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode base64 image data."
            )
            
        from .analyzer import analyze_item
        report = analyze_item(image_bytes)
        
        if not report.get("success", False):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": report.get("error", "Unknown error during analysis.")}
            )
            
        return report
        
    except Exception as e:
        logger.error(f"Error in /analyze-base64 upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def serve_frontend_page():
    """
    Serves the modern single-page dashboard.
    Loads dynamically from the frontend directory for immediate reflection of hot changes.
    """
    backend_dir = os.path.dirname(__file__) or '.'
    project_root = os.path.dirname(backend_dir)
    frontend_path = os.path.join(project_root, 'frontend', 'index.html')
    
    if os.path.exists(frontend_path):
        try:
            with open(frontend_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read(), status_code=200)
        except Exception as e:
            logger.error(f"Error reading frontend template: {e}")
            raise HTTPException(status_code=500, detail="Error reading index.html template file.")
            
    # Fallback response in case index.html is missing
    return HTMLResponse(content="""
    <html>
        <head><title>Setup Error</title></head>
        <body style="font-family: sans-serif; text-align: center; padding: 50px; background: #0f172a; color: #f1f5f9;">
            <h1 style="color: #f43f5e;">Template Not Found</h1>
            <p>Could not locate the frontend/index.html file.</p>
            <p>Please ensure your directories match the specification.</p>
        </body>
    </html>
    """, status_code=404)
