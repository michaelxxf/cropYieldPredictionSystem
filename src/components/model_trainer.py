import pandas as pd 
import numpy as np 
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from logger import logging 
from exception import CustomException 
from dataclasses import dataclass
from utils import save_function 
from utils import model_performance 
from sklearn.metrics import r2_score

@dataclass 
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("dataset", "model.pkl")


class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array): 
        try: 
            logging.info("Segregating the dependent and independent variables")
            X_train, y_train, X_test, y_test = (train_array[:, :-1], 
                                                train_array[:,-1], 
                                                test_array[:, :-1], 
                                                test_array[:,-1])
            
            # Using only RandomForest with better hyperparameters
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            logging.info("Training RandomForest model")
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"RandomForest R2 Score: {r2}")
            
            # Save the model
            save_function(file_path=self.model_trainer_config.trained_model_file_path, obj=model)
            logging.info("Model saved successfully")

        except Exception as e: 
            logging.info("Error occurred during model training")
            raise CustomException(e,sys)