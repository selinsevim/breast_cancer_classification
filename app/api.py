from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json

# Create a FastAPI instance
app = FastAPI()

model_path = r"model/model.joblib"
# Load the scaler and model
scaler, model = joblib.load(model_path)

class CancerModelInput(BaseModel):
    radius_mean: float = Field(..., example=14.2)
    texture_mean: float = Field(..., example=20.1)
    perimeter_mean: float = Field(..., example=95.0)
    area_mean: float = Field(..., example=700.0)
    smoothness_mean: float = Field(..., example=0.1)
    compactness_mean: float = Field(..., example=0.2)
    concavity_mean: float = Field(..., example=0.18)
    concave_points_mean: float = Field(..., example=0.1)
    symmetry_mean: float = Field(..., example=0.2)
    fractal_dimension_mean: float = Field(..., example=0.06)
    
    radius_se: float = Field(..., example=0.5)
    texture_se: float = Field(..., example=1.0)
    perimeter_se: float = Field(..., example=3.5)
    area_se: float = Field(..., example=50.0)
    smoothness_se: float = Field(..., example=0.005)
    compactness_se: float = Field(..., example=0.02)
    concavity_se: float = Field(..., example=0.03)
    concave_points_se: float = Field(..., example=0.015)
    symmetry_se: float = Field(..., example=0.02)
    fractal_dimension_se: float = Field(..., example=0.004)
    
    radius_worst: float = Field(..., example=17.0)
    texture_worst: float = Field(..., example=28.0)
    perimeter_worst: float = Field(..., example=120.0)
    area_worst: float = Field(..., example=900.0)
    smoothness_worst: float = Field(..., example=0.15)
    compactness_worst: float = Field(..., example=0.3)
    concavity_worst: float = Field(..., example=0.25)
    concave_points_worst: float = Field(..., example=0.15)
    symmetry_worst: float = Field(..., example=0.3)
    fractal_dimension_worst: float = Field(..., example=0.08)
    
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return f.read()
    
# Define a route for breast cancer classification
@app.post("/predict/")
def predict_cancer(input_data: CancerModelInput):
    # Create dataframe from inputs
    input_data = pd.DataFrame([input_data.dict()])
    scaled = scaler.transform(input_data)
    
    classification = model.predict(scaled)
    classification_label = "Benign" if classification == 0 else "Malignant"

    return {"result": classification_label}