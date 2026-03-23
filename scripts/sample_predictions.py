import pandas as pd 
import requests
from pathlib import Path

root_path = Path(__file__).parent.parent.parent 
data_path = root_path/"data"/"raw"/"swiggy.csv"
predict_url = "http://127.0.0.1:8000/predict"
