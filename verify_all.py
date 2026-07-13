import os
import sys

def verify_pipeline():
    print("=== STARTING PIPELINE VALIDATION ===")
    
    project_root = os.path.dirname(__file__) or '.'
    backend_dir = os.path.join(project_root, 'backend')
    
    # Add to path to allow imports
    sys.path.insert(0, project_root)
    
    # 1. Run local model training to generate price_model.joblib
    print("\n--- 1. Testing Local Model Training Pipeline ---")
    try:
        from backend.train_model import train_pricing_model
        train_pricing_model()
        print("PASS: Tabular pricing model successfully trained and serialized.")
    except Exception as e:
        print(f"FAIL: Tabular training failed: {e}")
        return False
        
    # 2. Check model loading
    print("\n--- 2. Testing Serialization & Loading ---")
    model_path = os.path.join(backend_dir, 'price_model.joblib')
    if not os.path.exists(model_path):
        print(f"FAIL: price_model.joblib was not created at {model_path}")
        return False
        
    try:
        import joblib
        import pandas as pd
        model = joblib.load(model_path)
        print("PASS: joblib successfully loaded model pipeline.")
        
        # Run dummy prediction
        test_df = pd.DataFrame([{
            'category': 'Car',
            'brand_tier': 3,
            'age_years': 3.5,
            'wear_score': 1.5
        }])
        pred = model.predict(test_df)[0]
        print(f"PASS: Regressor test prediction residual ratio: {pred*100:.2f}% value retained.")
    except Exception as e:
        print(f"FAIL: Model loading or inference test failed: {e}")
        return False
        
    # 3. Test YOLO category mapping
    print("\n--- 3. Testing Category Mapping ---")
    try:
        from backend.analyzer import map_yolo_to_app_category
        test_classes = {
            'car': 'Car',
            'motorcycle': 'Bike',
            'cell phone': 'Phone',
            'laptop': 'Laptop',
            'chair': 'Furniture',
            'person': 'Person',
            'banana': 'Other'
        }
        for cls, expected in test_classes.items():
            mapped = map_yolo_to_app_category(cls)
            assert mapped == expected, f"Expected {cls} -> {expected}, got {mapped}"
        print("PASS: YOLO category map logic is correct.")
    except Exception as e:
        print(f"FAIL: Category mapping failed: {e}")
        return False
        
    # 4. Check environmental vars config
    print("\n--- 4. Checking config templates ---")
    env_example = os.path.join(project_root, '.env.example')
    env_real = os.path.join(project_root, '.env')
    if os.path.exists(env_example) or os.path.exists(env_real):
        print("PASS: Configuration file (.env or .env.example) is present.")
    else:
        print("FAIL: Configuration file (.env or .env.example) is missing.")
        return False
        
    print("\n=== PIPELINE VALIDATION SUCCESSFUL ===")
    return True

if __name__ == '__main__':
    success = verify_pipeline()
    sys.exit(0 if success else 1)
