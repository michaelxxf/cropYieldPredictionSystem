import sys 
import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exception import CustomException 
from logger import logging 
from utils import load_obj

class PredictPipeline: 
    def __init__(self) -> None:
        self.model = None
        try:
            model_path = os.path.join('dataset', 'model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, features): 
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
                
            prediction = self.model.predict(features)
            # Ensure prediction is valid
            if isinstance(prediction, (np.ndarray, list)) and len(prediction) > 0:
                return prediction
            raise ValueError("Invalid prediction result")
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise Exception(f"Error in prediction pipeline: {str(e)}")
        
class CustomData: 
    def __init__(self, N:float, 
                 P:float, 
                 K:float, 
                 temperature:float,
                 humidity:float,
                 ph:float,
                 rainfall:float): 
         self.N = N
         self.P = P
         self.K = K
         self.temperature = temperature
         self.humidity = humidity
         self.ph = ph  # Input as 'ph'
         self.rainfall = rainfall
    
    def get_data_as_dataframe(self): 
         try: 
              custom_data_input_dict = {
                   'N': self.N,
                   'P': self.P,
                   'K': self.K,
                   'temperature': self.temperature,
                   'humidity': self.humidity,
                   'pH': self.ph,  # Convert to 'pH' for model
                   'rainfall': self.rainfall
              }
              df = pd.DataFrame([custom_data_input_dict])
              print(f"Created DataFrame with columns: {df.columns.tolist()}")  # Debug log
              return df
         except Exception as e:
              raise Exception(f"Error in creating DataFrame: {str(e)}")


