import sys 
import os 

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.preprocessing import *

input_path = r'data/raw/data.csv'
output_path = r'data/preprocessed/preprocessed.csv'

df = read_csv(input_path)
df_processed = prepare_columns(df)
df_processed.to_csv(output_path)
