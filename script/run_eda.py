import sys 
import os 

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.preprocessing import *

input_path = r'data/raw/data.csv'
df = read_csv(input_path)
inspect_data(df)
distribution_graph(df)
correlation_analysis(df)
scatter_matrix(df)
distribution_graph(df)
box_plot(df)
correlation_matrix(df)