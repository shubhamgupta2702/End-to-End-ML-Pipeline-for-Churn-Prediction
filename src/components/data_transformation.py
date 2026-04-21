import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception.exception import CustomException
from src.logger.logger import logger
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
  preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")
  
class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()
    
  def get_data_transformer_object(self):
    '''
    This function is responsible for Data Transformation.
    '''
    try:
      numerical_features = ['tenure','MonthlyCharges','TotalCharges']
      categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
      
      cat_pipeline = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('ohe', OneHotEncoder(handle_unknown='ignore'))
      ])

      num_pipeline = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())
      ])
      
      logger.info("Numerical and Categorical features transformed.")
      
      preprocesser = ColumnTransformer(
      transformers=[
      ('num', num_pipeline, numerical_features),
      ('cat', cat_pipeline, categorical_features)
      ], remainder='drop'
      )
      
      logger.info("Preprocesser using Column Transformer Done.")
      
      return preprocesser
    
    except Exception as e:
      logger.error("Error in Data Transformation")
      raise CustomException(e,sys)
    
  def initiate_data_transformation(self, train_path, test_path):
    try:
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)
      logger.info("Reading train and test data completed")
      
      train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
      test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
      logger.info("Changed non-numeric columns into numeric columns")
      
      train_df = train_df.dropna(subset=['TotalCharges'])
      test_df = test_df.dropna(subset=['TotalCharges'])
      logger.info("Dropped rows with null TotalCharges")
      
      
      logger.info("Obtaining preprocessing object")
      preprocessing_obj = self.get_data_transformer_object()
      
      target_column = 'Churn'
      numerical_features = ['tenure','MonthlyCharges','TotalCharges']
      
      input_feature_train_df = train_df.drop(columns=[target_column])
      target_feature_train_df = train_df[target_column]
      
      input_feature_test_df = test_df.drop(columns=[target_column])
      target_feature_test_df = test_df[target_column]
      
      target_mapping = {'No': 0, 'Yes': 1}

      target_feature_train_df = target_feature_train_df.map(target_mapping)
      target_feature_test_df = target_feature_test_df.map(target_mapping)
      
      
      logger.info(f"Applying Preprocessing object on training dataframe and testing dataframe.")
      
      input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
      
      train_arr = np.c_[
        input_feature_train_arr, np.array(target_feature_train_df)
        ]
      test_arr = np.c_[
        input_feature_test_arr, np.array(target_feature_test_df)
        ]
      
      logger.info("Combined features + encoded target")

      logger.info(f"Saved preprocessing object.")
      
      save_object(
        file_path = self.data_transformation_config.preprocessor_obj_path,
        obj = preprocessing_obj
      )
      
      return (
        train_arr, test_arr,
        self.data_transformation_config.preprocessor_obj_path
      )
      
    except Exception as e:
      logger.error("Error in initiate_data_transformation")
      raise CustomException(e, sys)