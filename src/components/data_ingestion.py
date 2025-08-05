import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import logging
from exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    Dataset_folder: str = 'dataset'
    train_data_filename: str = 'train.csv'
    test_data_filename: str = 'test.csv'
    raw_data_folder: str = 'database/data'
    raw_data_filename: str = 'crop_yield.csv'

    @property
    def train_data_path(self):
        return os.path.join(self.Dataset_folder, self.train_data_filename)

    @property
    def test_data_path(self):
        return os.path.join(self.Dataset_folder, self.test_data_filename)

    @property
    def raw_data_path(self):
        return os.path.join(self.raw_data_folder, self.raw_data_filename)

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def create_sample_data(self):
        """Create a sample dataset for testing"""
        data = {
            'N': [90, 85, 60, 74, 78, 85, 80, 70, 65, 75],
            'P': [42, 58, 55, 35, 42, 32, 62, 45, 38, 40],
            'K': [43, 41, 44, 40, 41, 45, 43, 40, 42, 44],
            'temperature': [24.5, 26.2, 23.8, 25.1, 24.7, 25.4, 23.9, 24.8, 25.6, 24.3],
            'humidity': [80, 85, 83, 87, 82, 86, 84, 81, 88, 85],
            'pH': [6.5, 6.2, 6.8, 6.4, 6.3, 6.6, 6.7, 6.5, 6.4, 6.3],
            'rainfall': [202.9, 190.4, 98.8, 110.2, 125.6, 142.3, 177.8, 156.4, 145.2, 167.9],
            'Production_in_tons': [5.2, 4.8, 6.1, 4.5, 5.8, 5.1, 4.2, 5.4, 6.2, 4.9]
        }
        return pd.DataFrame(data)

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            # Create sample data
            df = self.create_sample_data()
            logging.info('Sample dataset created')

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            os.makedirs(self.ingestion_config.Dataset_folder, exist_ok=True)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Train and test split completed')
            logging.info('Ingestion of Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error('Exception occurred during data ingestion')
            raise CustomException(e, sys)
