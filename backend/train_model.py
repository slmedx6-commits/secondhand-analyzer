import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

def generate_synthetic_data(num_samples=2500, random_seed=42):
    """
    Generates a realistic synthetic dataset for second-hand items price depreciation.
    """
    np.random.seed(random_seed)
    
    categories = ['Car', 'Bike', 'House', 'Phone', 'Laptop', 'Furniture', 'Person', 'Other']
    brand_tiers = [1, 2, 3] # 1: Budget, 2: Mid-tier, 3: Luxury
    
    data = []
    for _ in range(num_samples):
        cat = np.random.choice(categories)
        brand = np.random.choice(brand_tiers, p=[0.4, 0.45, 0.15])
        wear = np.random.uniform(0.0, 10.0) # 0 is brand new, 10 is scrap
        
        # Age distribution depends heavily on category
        if cat == 'House':
            age = np.random.exponential(15.0)
            age = np.clip(age, 0, 80)
        elif cat in ['Phone', 'Laptop']:
            age = np.random.uniform(0.1, 7.0)
        elif cat in ['Car', 'Bike']:
            age = np.random.uniform(0.1, 20.0)
        elif cat == 'Furniture':
            age = np.random.uniform(0.5, 25.0)
        else: # Person or Other
            age = np.random.uniform(0.0, 10.0)
            
        # Base depreciation calculation logic
        # 1. Age depreciation rate and minimum value clamp based on category
        if cat == 'House':
            age_deprec = -0.005 * age # Real estate structures/land generally appreciate slightly over time
            wear_deprec = (wear / 10.0) * 0.15 # Up to 15% loss for severe wear
            min_ratio = 0.60
            max_ratio = 1.50
        elif cat in ['Phone', 'Laptop']:
            age_deprec = 0.08 * age # Tech depreciates steadily but flattens out
            wear_deprec = (wear / 10.0) * 0.25 # Up to 25% loss for severe wear
            min_ratio = 0.15
            max_ratio = 0.95
        elif cat in ['Car', 'Bike']:
            age_deprec = 0.035 * age # Vehicles deprecate around 3.5% per year
            wear_deprec = (wear / 10.0) * 0.20 # Up to 20% loss for severe wear
            min_ratio = 0.30
            max_ratio = 0.95
        elif cat == 'Furniture':
            age_deprec = 0.02 * age # Furniture depreciates very slowly
            wear_deprec = (wear / 10.0) * 0.20 # Up to 20% loss for severe wear
            min_ratio = 0.25
            max_ratio = 0.95
        else:
            age_deprec = 0.05 * age
            wear_deprec = (wear / 10.0) * 0.20
            min_ratio = 0.20
            max_ratio = 0.95
            
        # 2. Brand tier retention bonus (luxury retains more value)
        brand_bonus = 0.05 * (brand - 1)
        
        # Calculate retained ratio (percentage of original price remaining)
        retained_ratio = 0.95 - age_deprec - wear_deprec + brand_bonus
        
        # Add noise
        noise = np.random.normal(0, 0.03)
        retained_ratio += noise
        
        # Clip ratio to its specific bounds based on category
        retained_ratio = np.clip(retained_ratio, min_ratio, max_ratio)
        
        data.append({
            'category': cat,
            'brand_tier': brand,
            'age_years': round(age, 2),
            'wear_score': round(wear, 2),
            'depreciation_ratio': round(retained_ratio, 4)
        })
        
    return pd.DataFrame(data)

def train_pricing_model():
    """
    Trains the RandomForestRegressor pipeline and saves it.
    """
    print("Generating synthetic data for Second-Hand Item Pricing Model...")
    df = generate_synthetic_data()
    
    X = df[['category', 'brand_tier', 'age_years', 'wear_score']]
    y = df['depreciation_ratio']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['category'])
        ],
        remainder='passthrough'
    )
    
    # Model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1))
    ])
    
    print("Training Random Forest Regressor...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    
    predictions = model_pipeline.predict(X_test)
    mae = np.mean(np.abs(predictions - y_test))
    
    print(f"Model Training complete.")
    print(f"Train R^2 Score: {train_score:.4f}")
    print(f"Test R^2 Score: {test_score:.4f}")
    print(f"Mean Absolute Error (MAE) on Retained Price Ratio: {mae*100:.2f}%")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(__file__) or '.', exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__) or '.', 'price_model.joblib')
    
    joblib.dump(model_pipeline, model_path)
    print(f"Model successfully saved to {model_path}")
    
    # Print sample predictions
    print("\nSample predictions:")
    test_samples = pd.DataFrame([
        {'category': 'Phone', 'brand_tier': 3, 'age_years': 2.0, 'wear_score': 2.0},  # Premium phone, 2 yrs old, minor wear
        {'category': 'Car', 'brand_tier': 2, 'age_years': 5.0, 'wear_score': 4.5},    # Mid-tier car, 5 yrs old, moderate wear
        {'category': 'Laptop', 'brand_tier': 1, 'age_years': 4.0, 'wear_score': 7.0}, # Budget laptop, 4 yrs old, heavy wear
    ])
    predicted_ratios = model_pipeline.predict(test_samples)
    for i, row in test_samples.iterrows():
        print(f"Category: {row['category']}, Brand Tier: {row['brand_tier']}, Age: {row['age_years']} yrs, Wear Score: {row['wear_score']}/10")
        print(f" -> Predicted Residual Value: {predicted_ratios[i]*100:.2f}% of original price")

if __name__ == '__main__':
    train_pricing_model()
