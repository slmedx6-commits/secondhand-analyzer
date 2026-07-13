# AI Second-Hand Item Analyzer & Age Estimation

A hybrid, resource-optimized web application designed to run on consumer hardware (Intel Core i3, 8GB RAM, No GPU) under Windows. It inspects second-hand objects, detects physical wear, estimates age, and predicts residual price in Indian Rupees (₹) using a combination of local CPU models (YOLOv8 + scikit-learn RandomForestRegressor) and cloud visual appraisal (Gemini 3.5 Flash).

---

## 🛠️ Technology Stack & Machine Learning Pipelines

1. **Local Object Detection (YOLOv8-nano)**: Detects, highlights, and localizes the item in system memory, returning bounding-box annotated frames.
2. **Local Price Prediction (scikit-learn)**: A tabular Random Forest Regressor trained on synthetic transaction statistics that predicts depreciation ratios based on category, age, brand tier, and visual wear.
3. **Cloud Visual Inspection (Gemini 3.5 Flash)**: Pinpoints brand/model, estimated release year, details visual wear (scratches, dents, cracks, rust), and generates natural language reasoning.
4. **FastAPI Backend & Tailwind CSS UI**: Lightweight async service and responsive, glassmorphic dashboard.

---

## ⚙️ Project Structure

```text
secondhand-analyzer/
├── backend/
│   ├── __init__.py
│   ├── app.py            # FastAPI service endpoints
│   ├── analyzer.py       # YOLOv8 + Gemini + tabular pricing pipeline
│   ├── train_model.py    # Training & evaluation pipeline for pricing Regressor
│   └── price_model.joblib # Serialized RandomForestRegressor pipeline
├── frontend/
│   └── index.html        # Glassmorphic client UI
├── requirements.txt      # Optimized libraries for Windows CPU
├── .env.example          # Environment variables template
└── README.md             # This guide
```

---

## 🚀 Local Installation & Execution

### 1. Prerequisites
- Python 3.10+ installed on Windows.
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 2. Setup environment on Windows
Open PowerShell and navigate to the project directory:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Configure API Key
Duplicate `.env.example` as `.env`:
```powershell
copy .env.example .env
```
Open `.env` and fill in your `GEMINI_API_KEY`:
```env
GEMINI_API_KEY=AIzaSy...
```

### 4. Run the Application
Start the FastAPI server:
```powershell
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```
- **Self-Generating ML Model**: The backend will automatically trigger `train_model.py` on startup if `price_model.joblib` is missing, training your tabular price predictor in milliseconds.
- Open `http://127.0.0.1:8000` in your web browser.

---

## 📈 Windows 11/12 Low-RAM Pagefile Optimization

With only 8GB of RAM, loading PyTorch dependencies and models can sometimes trigger Windows Out-Of-Memory (OOM) exceptions. To prevent system crashes, configure a virtual memory pagefile to back system memory with SSD storage:

1. Press `Win + R`, type `sysdm.cpl`, and hit Enter to open **System Properties**.
2. Go to the **Advanced** tab. Under **Performance**, click **Settings...**.
3. Go to the **Advanced** tab in Performance Options, then under **Virtual memory**, click **Change...**.
4. Uncheck **Automatically manage paging file size for all drives**.
5. Select your C: drive, select **Custom size**, and enter:
   - **Initial size**: `8192` (8 GB)
   - **Maximum size**: `16384` (16 GB)
6. Click **Set**, then click **OK**, and restart your computer to apply.

---

## ☁️ Free Hosting Deployment Guide

### Deployment Option A: Hugging Face Spaces (Free Tier CPU)
1. Sign in to [Hugging Face](https://huggingface.co/) and click **New Space**.
2. Select **Docker** as the SDK (FastAPI works best via Docker for custom system dependencies like YOLOv8).
3. Select **Blank** docker template.
4. Add the following `Dockerfile` to the root:
   ```dockerfile
   FROM python:3.10-slim
   RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   RUN python backend/train_model.py
   EXPOSE 7860
   CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
5. Commit files to the Space. Under Space **Settings**, add your `GEMINI_API_KEY` to **Repository secrets**.

### Deployment Option B: Streamlit Community Cloud (Migrating to Streamlit UI)
To deploy on Streamlit Community Cloud:
1. Re-implement the UI elements using streamlit functions inside `app.py`.
2. Push your project to a GitHub repository.
3. Link your GitHub repo to [Streamlit Share](https://share.streamlit.io/).
4. Add your `GEMINI_API_KEY` under the app secrets menu.
