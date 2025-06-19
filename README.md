# AI-Powered Breast Cancer Prediction Assistant

This project is a web application that uses a XGBoost model to predict breast cancer diagnosis (Benign or Malignant) based on patient data. The backend is built with FastAPI, and the frontend is a simple HTML/CSS form that collects user inputs and displays the prediction result.

![AI-Powered Breast Cancer Prediction Assistant](static/assets/localhost.png)
![AI-Powered Breast Cancer Prediction Assistant Result](static/assets/benign.png)

---

## Features

- Predicts breast cancer diagnosis from input features using a pre-trained ML model.
- User-friendly web interface with form inputs for all required features.
- Displays prediction result on the same page without reloading.
- Dockerized for easy deployment.

---

## Project Structure

```bash
breast_cancer_prediction_ai_model/
├── app/
│ └── api.py # FastAPI backend
├── model/
│ └── model.joblib # Pre-trained ML model and scaler
│ └── calibrated_xgb_model.pkl
│ └── scaler.pkl
├── data/
│ └── preprocessed/
│       └──preprocessed.csv
│ └── raw/
│       └──data.csv
├── notebooks/
│ └── analysis.ipynb
├── script/
│ └── run_eda.py
│ └── run_preprocessing.py
│ └── run_training.py
├── src/
│ └── model/
│       └──model.py
│ └── preprocessing/
│       └──preprocessing.py
├── static/
│ ├── index.html # Frontend HTML page
│ ├── style.css # Styling for frontend
│ └── assets/ # Images used in the frontend
├── requirements.txt # Python dependencies
├── Dockerfile # Dockerfile to containerize the app
└── README.md # This file
└── docker_compose.yaml # This file
```

## Setup and Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/selinsevim/breast_cancer_classification
cd breast_cancer_prediction_ai_model
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI app

```bash
uvicorn app.api:app --reload
```

The app will be available at http://localhost:8000

### 5. Using Docker

Build the Docker image

```bash
docker build -t breast-cancer-app .
```

### 6. Run the Docker container

```bash
docker run -p 8000:8000 breast-cancer-app
```

Open http://localhost:8000 in your browser.

### 7. API Endpoint

POST /predict/ - Accepts JSON with breast cancer features and returns prediction result.

Example request body:

```bash
{
  "radius_mean": 14.2,
  "texture_mean": 20.1,
  "perimeter_mean": 95.0,
  "area_mean": 700.0,
  "smoothness_mean": 0.1,
  "compactness_mean": 0.2,
  "concavity_mean": 0.18,
  "concave_points_mean": 0.1,
  "symmetry_mean": 0.2,
  "fractal_dimension_mean": 0.06,
  "radius_se": 0.5,
  "texture_se": 1.0,
  "perimeter_se": 3.5,
  "area_se": 50.0,
  "smoothness_se": 0.005,
  "compactness_se": 0.02,
  "concavity_se": 0.03,
  "concave_points_se": 0.015,
  "symmetry_se": 0.02,
  "fractal_dimension_se": 0.004,
  "radius_worst": 17.0,
  "texture_worst": 28.0,
  "perimeter_worst": 120.0,
  "area_worst": 900.0,
  "smoothness_worst": 0.15,
  "compactness_worst": 0.3,
  "concavity_worst": 0.25,
  "concave_points_worst": 0.15,
  "symmetry_worst": 0.3,
  "fractal_dimension_worst": 0.08
}
```
