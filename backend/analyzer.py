import os
import io
import gc
import json
import base64
import logging
import warnings
from PIL import Image
import numpy as np
import pandas as pd
import joblib

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy loaded models to minimize RAM usage
_yolo_model = None
_pricing_model = None

def get_yolo_model():
    """
    Lazy load YOLOv8 model for object detection.
    """
    global _yolo_model
    if _yolo_model is None:
        logger.info("Loading YOLOv8 model...")
        from ultralytics import YOLO
        # yolov8n.pt is 6.2MB, extremely light and runs fast on CPU
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model

def get_pricing_model():
    """
    Lazy load local scikit-learn pricing model.
    """
    global _pricing_model
    if _pricing_model is None:
        model_path = os.path.join(os.path.dirname(__file__) or '.', 'price_model.joblib')
        if os.path.exists(model_path):
            logger.info("Loading pricing model...")
            _pricing_model = joblib.load(model_path)
        else:
            logger.warning(f"Pricing model not found at {model_path}. Auto-training may be required.")
    return _pricing_model

def map_yolo_to_app_category(yolo_class_name):
    """
    Maps YOLOv8 standard classes to our application categories.
    """
    yolo_class_name = yolo_class_name.lower()
    
    mapping = {
        # Cars
        'car': 'Car', 'truck': 'Car', 'bus': 'Car', 'van': 'Car',
        # Bikes
        'motorcycle': 'Bike', 'bicycle': 'Bike',
        # Phones
        'cell phone': 'Phone',
        # Laptops & tech
        'laptop': 'Laptop', 'tv': 'Laptop', 'keyboard': 'Laptop', 'mouse': 'Laptop',
        # Furniture
        'chair': 'Furniture', 'couch': 'Furniture', 'bed': 'Furniture', 
        'dining table': 'Furniture', 'bench': 'Furniture', 'sofa': 'Furniture',
        # Person
        'person': 'Person'
    }
    
    return mapping.get(yolo_class_name, 'Other')

def preprocess_and_resize(pil_image, max_size=1024):
    """
    Resizes image to keep aspect ratio but clamp max dimension to max_size.
    Reduces network traffic and API RAM consumption.
    """
    w, h = pil_image.size
    if max(w, h) > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))
        logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
        return pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return pil_image

def analyze_item(image_bytes: bytes) -> dict:
    """
    Processes the image and performs analysis.
    1. Local YOLOv8 for object localization and category mapping.
    2. Gemini 3.5 Flash for deep visual wear and brand/model identification.
    3. Local Tabular RandomForestRegressor for second-hand price prediction.
    """
    try:
        # Load PIL image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Immediately downscale the image to speed up local YOLOv8 processing and Gemini network transit.
        # 800px is optimal for visual condition inspection and object detection.
        pil_image = preprocess_and_resize(pil_image, max_size=800)
        
        # 1. Local YOLOv8 Object Detection
        yolo_model = get_yolo_model()
        yolo_results = yolo_model(pil_image, verbose=False)
        
        yolo_category = "Other"
        yolo_confidence = 0.0
        annotated_base64 = ""
        yolo_detections = []
        
        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
            result = yolo_results[0]
            boxes = result.boxes
            
            # Find the detection with the highest confidence
            highest_conf_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            top_box = boxes[highest_conf_idx]
            top_class_id = int(top_box.cls[0].cpu().item())
            top_class_name = result.names[top_class_id]
            
            yolo_category = map_yolo_to_app_category(top_class_name)
            yolo_confidence = float(top_box.conf[0].cpu().item()) * 100
            
            # Draw bounding boxes and encode to base64
            annotated_img = result.plot()
            # Convert numpy array to PIL Image
            annotated_pil = Image.fromarray(annotated_img[:, :, ::-1]) # Convert BGR to RGB
            # Crop to max size
            annotated_pil = preprocess_and_resize(annotated_pil)
            
            buffered = io.BytesIO()
            annotated_pil.save(buffered, format="JPEG", quality=85)
            annotated_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # List all detections for UI display
            for box in boxes:
                cls_id = int(box.cls[0].cpu().item())
                name = result.names[cls_id]
                conf = float(box.conf[0].cpu().item())
                yolo_detections.append({
                    "label": name,
                    "confidence": round(conf * 100, 1),
                    "mapped_category": map_yolo_to_app_category(name)
                })
        else:
            # No YOLO detections, encode resized image
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            annotated_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            logger.info("No objects detected locally by YOLOv8.")
            
        # 2. Prepare Gemini 3.5 Flash Visual-Language Model Analysis
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured. Please set the API key in settings or environment.")
            
        genai.configure(api_key=gemini_api_key)
        
        # System instructions and prompt forcing structured JSON
        system_prompt = (
            "You are an expert appraiser and physical condition inspector. You analyze images of second-hand "
            "items and output structured appraisal data in raw JSON format.\n"
            "The current calendar year is 2026. Use this reference year to calculate the age of the item.\n"
            "Analyze the image and estimate the physical condition, brand/model, and release year.\n"
            "Estimate original price in Indian Rupees (INR, ₹). Make sure it reflects the realistic retail cost in India "
            "when this model was new.\n"
            "Rate wear_score on a float scale from 0.0 (absolutely brand new, in box) to 10.0 (completely broken/waste).\n"
            "Return a JSON object conforming exactly to the following schema:\n"
            "{\n"
            '  "brand": "string (brand name, e.g. Apple, Maruti Suzuki, IKEA, Samsung)",\n'
            '  "model": "string (exact model name, e.g. iPhone 13 Pro, Swift VXI, Poäng Chair)",\n'
            '  "estimated_mfg_year": "string (year of manufacture, e.g. 2021)",\n'
            '  "age_years": float (estimated age in years relative to 2026, e.g. 5.0),\n'
            '  "brand_tier": int (1 = budget, 2 = mid-range/popular, 3 = premium/luxury/exotic),\n'
            '  "original_price_in_rupees": float (average retail price when new in Indian Rupees, e.g. 119900.0),\n'
            '  "scratches_severity": "None" | "Minor" | "Moderate" | "Severe",\n'
            '  "scratches_details": "string (brief visual details)",\n'
            '  "dents_severity": "None" | "Minor" | "Moderate" | "Severe",\n'
            '  "dents_details": "string (brief visual details)",\n'
            '  "cracks_severity": "None" | "Minor" | "Moderate" | "Severe",\n'
            '  "cracks_details": "string (brief visual details)",\n'
            '  "rust_severity": "None" | "Minor" | "Moderate" | "Severe",\n'
            '  "rust_details": "string (brief visual details)",\n'
            '  "other_wear_details": "string (brief comments on faded paint, dust, leather wear, etc.)",\n'
            '  "wear_score": float (0.0 to 10.0 representing visual wear score where higher means more damaged),\n'
            '  "explanation": "string (Explainable AI: detail why you estimated this year, brand, and condition based on visual cues in the image)"\n'
            "}"
        )
        
        # Configure model parameters
        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.2
        }
        
        logger.info("Invoking Gemini 3.5 Flash...")
        model = genai.GenerativeModel('gemini-3.5-flash', generation_config=generation_config)
        
        response = model.generate_content(
            contents=[system_prompt, pil_image]
        )
        
        gemini_result = {}
        try:
            gemini_result = json.loads(response.text)
            logger.info("Successfully parsed JSON response from Gemini API.")
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse Gemini response as JSON: {response.text}")
            # Robust fallback on parsing error
            gemini_result = {
                "brand": "Generic/Unknown",
                "model": "Unknown Item",
                "estimated_mfg_year": "2022",
                "age_years": 4.0,
                "brand_tier": 2,
                "original_price_in_rupees": 15000.0,
                "scratches_severity": "Minor",
                "scratches_details": "Undetected via raw parse",
                "dents_severity": "None",
                "dents_details": "Undetected",
                "cracks_severity": "None",
                "cracks_details": "Undetected",
                "rust_severity": "None",
                "rust_details": "Undetected",
                "other_wear_details": "Undetected",
                "wear_score": 3.0,
                "explanation": f"API returned non-JSON format. Fallback activated. Raw response: {response.text[:200]}..."
            }
            
        # 3. Local Machine Learning - Tabular Regressor for Second-Hand Price
        pricing_model = get_pricing_model()
        
        # Resolve category (use YOLO if confident, otherwise trust Gemini's visual class determination)
        final_category = yolo_category
        if yolo_category == 'Other' or yolo_confidence < 40.0:
            # Categorize from Gemini's output
            detected_brand_model = f"{gemini_result.get('brand', '')} {gemini_result.get('model', '')}".lower()
            gemini_cat = "Other"
            for keyword, mapped in [('car', 'Car'), ('auto', 'Car'), ('suv', 'Car'), ('truck', 'Car'), ('honda', 'Car'), ('suzuki', 'Car'), ('hyundai', 'Car'),
                                    ('bike', 'Bike'), ('cycle', 'Bike'), ('motorcycle', 'Bike'),
                                    ('phone', 'Phone'), ('samsung', 'Phone'), ('iphone', 'Phone'), ('mobile', 'Phone'),
                                    ('laptop', 'Laptop'), ('macbook', 'Laptop'), ('dell', 'Laptop'), ('hp', 'Laptop'), ('lenovo', 'Laptop'),
                                    ('house', 'House'), ('apartment', 'House'), ('villa', 'House'),
                                    ('chair', 'Furniture'), ('sofa', 'Furniture'), ('table', 'Furniture'), ('bed', 'Furniture'), ('desk', 'Furniture')]:
                if keyword in detected_brand_model:
                    gemini_cat = mapped
                    break
            final_category = gemini_cat if gemini_cat != "Other" else yolo_category
            
        brand_tier = int(gemini_result.get('brand_tier', 2))
        age_years = float(gemini_result.get('age_years', 3.0))
        wear_score = float(gemini_result.get('wear_score', 3.0))
        original_price = float(gemini_result.get('original_price_in_rupees', 10000.0))
        
        # Predict depreciated value ratio using scikit-learn RandomForestRegressor model
        depreciation_ratio = 0.50 # Default fallback
        if pricing_model is not None:
            try:
                input_df = pd.DataFrame([{
                    'category': final_category,
                    'brand_tier': brand_tier,
                    'age_years': age_years,
                    'wear_score': wear_score
                }])
                depreciation_ratio = float(pricing_model.predict(input_df)[0])
                logger.info(f"Local ML Regressor Depreciation Ratio: {depreciation_ratio*100:.2f}%")
            except Exception as pe:
                logger.error(f"Error predicting with local ML model: {pe}")
        else:
            logger.warning("Local pricing model is not loaded. Using fallback depreciation rules.")
            # Fallback mathematical rule
            deprec_rate = 0.08 if final_category in ['Car', 'Bike'] else 0.18 if final_category in ['Phone', 'Laptop'] else 0.05
            depreciation_ratio = max(0.05, 0.95 - (deprec_rate * age_years) - (wear_score / 10.0) * 0.40)
            
        estimated_secondhand_price = original_price * depreciation_ratio
        
        # 4. Construct complete response payload
        results_payload = {
            "success": True,
            "yolo_category": yolo_category,
            "yolo_confidence": round(yolo_confidence, 1),
            "yolo_detections": yolo_detections,
            "final_category": final_category,
            "brand": gemini_result.get("brand", "Generic"),
            "model": gemini_result.get("model", "Unknown Model"),
            "estimated_mfg_year": gemini_result.get("estimated_mfg_year", "Unknown"),
            "age_years": round(age_years, 1),
            "brand_tier": brand_tier,
            "original_price_in_rupees": round(original_price, 2),
            "depreciation_ratio": round(depreciation_ratio, 4),
            "estimated_price_in_rupees": round(estimated_secondhand_price, 2),
            "wear_score": round(wear_score, 1),
            "anomalies": {
                "scratches": {
                    "severity": gemini_result.get("scratches_severity", "None"),
                    "details": gemini_result.get("scratches_details", "No visible scratches.")
                },
                "dents": {
                    "severity": gemini_result.get("dents_severity", "None"),
                    "details": gemini_result.get("dents_details", "No visible dents.")
                },
                "cracks": {
                    "severity": gemini_result.get("cracks_severity", "None"),
                    "details": gemini_result.get("cracks_details", "No visible cracks.")
                },
                "rust": {
                    "severity": gemini_result.get("rust_severity", "None"),
                    "details": gemini_result.get("rust_details", "No visible rust.")
                },
                "other": gemini_result.get("other_wear_details", "No other structural wear detected.")
            },
            "explanation": gemini_result.get("explanation", "No analysis provided."),
            "annotated_image": annotated_base64
        }
        
        return results_payload
        
    except Exception as e:
        logger.error(f"Global exception in analysis module: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Strict memory reclamation
        gc.collect()
