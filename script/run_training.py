import sys 
import os 

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.model import *

input_path = r'/data/preprocessed/preprocessed.csv'

df = read_csv(input_path)
X,y = prepare_data(df)
fold_metrics = cross_validation(X,y)
aggregate_metrics(fold_metrics)

model, scaler, calibrated_model = full_training(df)
save_models(model, calibrated_model, scaler)