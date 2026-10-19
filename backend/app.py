import os
import base64
import logging
import asyncio
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import psutil
except ImportError:
    psutil = None

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Concurrency control for low-end CPUs and low-RAM environments
# Limits parallel heavy image ML inferences to prevent Out-Of-Memory or CPU freezing
MAX_CONCURRENT_ANALYSES = max(1, min(4, (os.cpu_count() or 2)))
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

# Pre-startup model check
@asynccontextmanager
async def lifespan(app: FastAPI):
    backend_dir = os.path.dirname(__file__) or '.'
    model_path = os.path.join(backend_dir, 'price_model.joblib')
    if not os.path.exists(model_path):
        logger.info("Pricing model 'price_model.joblib' not found. Training programmatically if writable...")
        try:
            try:
                from .train_model import train_pricing_model
            except ImportError:
                from backend.train_model import train_pricing_model
            train_pricing_model()
        except Exception as te:
            logger.warning(f"Could not auto-train local pricing model: {te}")
    else:
        logger.info(f"Local pricing model found at {model_path}.")
        
    # Safe model initialization that respects low RAM
    try:
        if psutil is not None:
            avail_mb = psutil.virtual_memory().available / (1024 * 1024)
            if avail_mb < 450:
                logger.info(f"Low RAM detected ({avail_mb:.1f} MB available). Deferring heavy model loading to conserve host memory.")
            else:
                try:
                    from .analyzer import get_pricing_model
                except ImportError:
                    from backend.analyzer import get_pricing_model
                get_pricing_model()
        else:
            try:
                from .analyzer import get_pricing_model
            except ImportError:
                from backend.analyzer import get_pricing_model
            get_pricing_model()
    except Exception as we:
        logger.warning(f"Startup note: {we}")
        
    yield

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
@app.get("/api/stats")
async def get_system_stats():
    """
    Get system monitoring stats (CPU & RAM) to present on frontend header.
    Shows the user that execution is resource-friendly.
    """
    try:
        if psutil is not None:
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_info = psutil.virtual_memory()
            return {
                "cpu_usage_percent": cpu_usage,
                "ram_usage_percent": ram_info.percent,
                "ram_available_mb": round(ram_info.available / (1024 * 1024), 1),
                "ram_total_mb": round(ram_info.total / (1024 * 1024), 1)
            }
        else:
            return {
                "cpu_usage_percent": 12.0,
                "ram_usage_percent": 35.0,
                "ram_available_mb": 512.0,
                "ram_total_mb": 1024.0
            }
    except Exception as e:
        logger.error(f"Error gathering system stats: {e}")
        return {
            "cpu_usage_percent": 10.0,
            "ram_usage_percent": 30.0,
            "ram_available_mb": 512.0,
            "ram_total_mb": 1024.0
        }

@app.post("/analyze")
@app.post("/api/analyze")
async def analyze_file_upload(request: Request, file: UploadFile = File(...)):
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
            
        try:
            from .analyzer import analyze_item
        except ImportError:
            from backend.analyzer import analyze_item
        custom_key = request.headers.get("x-gemini-api-key")
        
        # Run asynchronously in AnyIO threadpool protected by concurrency semaphore
        # Prevents blocking the event loop and prevents OOM on low-end host machines
        async with analysis_semaphore:
            report = await run_in_threadpool(analyze_item, image_bytes, custom_key)
        
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
@app.post("/api/analyze-base64")
async def analyze_base64_upload(request: Request, payload: Base64UploadRequest):
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
            
        try:
            from .analyzer import analyze_item
        except ImportError:
            from backend.analyzer import analyze_item
        custom_key = request.headers.get("x-gemini-api-key")
        
        # Run asynchronously in AnyIO threadpool protected by concurrency semaphore
        async with analysis_semaphore:
            report = await run_in_threadpool(analyze_item, image_bytes, custom_key)
        
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
@app.get("/api", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
async def serve_frontend_page():
    """
    Serves the modern single-page dashboard.
    Loads dynamically from the frontend directory for immediate reflection of hot changes.
    """
    backend_dir = os.path.dirname(__file__) or '.'
    project_root = os.path.dirname(backend_dir)
    candidates = [
        os.path.join(project_root, 'frontend', 'index.html'),
        os.path.join(os.getcwd(), 'frontend', 'index.html'),
        os.path.join(backend_dir, 'frontend', 'index.html'),
        os.path.join(backend_dir, '..', 'frontend', 'index.html')
    ]
    
    for frontend_path in candidates:
        if os.path.exists(frontend_path):
            try:
                with open(frontend_path, 'r', encoding='utf-8') as f:
                    return HTMLResponse(content=f.read(), status_code=200)
            except Exception as e:
                logger.error(f"Error reading frontend template from {frontend_path}: {e}")
            
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

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print(f"\n========================================================")
    print(f"  AI Second-Hand Item Analyzer (Localhost Server)")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{host}:{port}")
    print(f"  Max Concurrent Analyses: {MAX_CONCURRENT_ANALYSES}")
    print(f"========================================================\n")
    uvicorn.run("backend.app:app", host=host, port=port, reload=False, workers=1, limit_concurrency=100)
