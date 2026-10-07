import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    HealthResponse,
    PredictResponse,
    HistoryItem,
    StatisticsResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModelInfoResponse
)
from src.inference.predictor import WastePredictor
from database.database import DatabaseManager
from src.utils.config_loader import get_config
from src.utils.logger import logger

# Initialize application components
config = get_config()
db = DatabaseManager()
predictor = WastePredictor()
# Preload verified feedback cache into predictor memory
try:
    predictor.feedback_hash_cache.update(db.get_all_cached_feedback())
    logger.info(f"Preloaded {len(predictor.feedback_hash_cache)} verified feedback items into predictor memory.")
except Exception as e:
    logger.warning(f"Could not preload feedback cache: {e}")

app = FastAPI(
    title=config["api"]["title"],
    version=config["api"]["version"],
    description="Production Computer Vision & Circular Economy Recommendation System"
)

# Enable CORS for local and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = Path(config["paths"]["uploads_dir"])
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# Mount static frontend and uploads
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "AI Circular Economy System API is live. Visit /docs for OpenAPI specs."})


@app.get("/style.css", include_in_schema=False)
async def serve_css():
    css_path = FRONTEND_DIR / "style.css"
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS not found")


@app.get("/script.js", include_in_schema=False)
async def serve_js():
    js_path = FRONTEND_DIR / "script.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS not found")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health and model readiness probe."""
    return HealthResponse(
        status="healthy",
        model_name=predictor.model_name,
        is_model_trained=predictor.is_trained,
        version=config["api"]["version"],
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_waste(
    file: UploadFile = File(...)
):
    """
    Classifies waste image, generates confidence, Grad-CAM heatmap,
    circular recommendation, and persists transaction to SQLite database.
    """
    # 1. Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    allowed_exts = config["api"].get("allowed_extensions", [".jpg", ".jpeg", ".png", ".webp"])
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}'. Allowed formats: {allowed_exts}"
        )

    # 2. Read bytes and enforce size limits
    max_size = config["api"].get("max_upload_size_bytes", 10485760)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size limit of {max_size // (1024*1024)}MB."
        )

    # 3. Save file securely (prevent path traversal)
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    saved_path = UPLOADS_DIR / safe_filename
    with open(saved_path, "wb") as f:
        f.write(contents)

    # 4. Execute AI prediction pipeline
    try:
        result = predictor.predict(contents, include_gradcam=True)
    except Exception as e:
        logger.error(f"Prediction failed on {file.filename}: {e}")
        # Clean up uploaded file if processing failed
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image decoding or model inference failed: {str(e)}"
        )

    # Check if exact image has verified human feedback cache
    import hashlib
    img_hash = hashlib.sha256(contents).hexdigest()
    cached_correction = db.get_cached_feedback_by_hash(img_hash)
    if cached_correction:
        logger.info(f"Image matches verified human feedback memory: overriding with '{cached_correction}'")
        result["prediction"]["class"] = cached_correction
        result["prediction"]["confidence"] = 0.9999
        # Update probabilities
        for k in result["prediction"]["all_probabilities"]:
            result["prediction"]["all_probabilities"][k] = 0.0001
        result["prediction"]["all_probabilities"][cached_correction] = 0.9999
        # Update recommendation
        rec_override = predictor.recommendation_engine.evaluate(cached_correction, 0.9999)
        rec_override["confidence_warning"] = "Verified by Prior User Feedback"
        result["recommendation"] = rec_override

    # 5. Persist prediction into SQLite
    pred_data = result["prediction"]
    rec_data = result["recommendation"]
    
    # Store relative web-accessible path for persistence
    db_img_rel_path = f"/uploads/{safe_filename}"
    pred_id = db.insert_prediction(
        predicted_class=pred_data["class"],
        confidence=pred_data["confidence"],
        recommended_action=rec_data["recommended_action"],
        circular_strategy=rec_data["circular_strategy"],
        environmental_priority=rec_data["environmental_priority"],
        image_path=db_img_rel_path
    )

    return PredictResponse(
        id=pred_id,
        prediction=pred_data,
        recommendation=rec_data,
        explainability=result["explainability"]
    )


@app.get("/history", response_model=List[HistoryItem])
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """Fetches paginated prediction audit history."""
    return db.get_history(limit=limit, offset=offset)


@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """Computes real-time distribution statistics from SQLite."""
    return db.get_statistics()


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackRequest):
    """Submits human verification or correction, actively updating the model and dataset."""
    import shutil
    import hashlib

    existing = db.get_prediction_by_id(payload.prediction_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction record not found.")

    target_class = payload.corrected_class if payload.corrected_class else existing["predicted_class"]

    success = db.update_feedback(
        prediction_id=payload.prediction_id,
        user_feedback=payload.user_feedback,
        corrected_class=payload.corrected_class
    )

    # Process image learning and hash caching
    img_rel = existing.get("image_path")
    if img_rel:
        clean_rel = img_rel.lstrip("/").replace("uploads/", "")
        full_path = UPLOADS_DIR / clean_rel
        if full_path.exists():
            with open(full_path, "rb") as f_img:
                img_bytes = f_img.read()
            # 1. Cache exact hash so re-uploads instantly output correct class
            h = hashlib.sha256(img_bytes).hexdigest()
            db.cache_image_feedback(h, target_class)

            # 2. Add sample permanently to dataset/train/<target_class>/
            target_dir = Path(config["paths"]["train_dir"]) / target_class
            target_dir.mkdir(parents=True, exist_ok=True)
            new_train_file = target_dir / f"feedback_{uuid.uuid4().hex[:8]}.jpg"
            shutil.copy2(full_path, new_train_file)
            logger.info(f"Saved feedback image to training dataset: {new_train_file}")

            # 3. Actively adapt the neural network weights immediately!
            try:
                predictor.learn_sample(img_bytes, target_class, steps=5)
            except Exception as e:
                logger.warning(f"Active online learning failed: {e}")

    if success:
        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded! The system has actively learned that this sample is {target_class.upper()}."
        )
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update record.")


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Returns architecture details and benchmark comparison table."""
    csv_path = Path(config["paths"]["comparison_csv"])
    comparison_data = []
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            comparison_data = df.to_dict(orient="records")
        except Exception as e:
            logger.warning(f"Failed to read comparison CSV: {e}")

    return ModelInfoResponse(
        active_model=config["model"]["name"],
        classes=config["classes"],
        num_classes=config["model"]["num_classes"],
        confidence_thresholds={
            "high": config["inference"]["confidence_high_threshold"],
            "moderate": config["inference"]["confidence_moderate_threshold"]
        },
        model_comparison=comparison_data
    )
