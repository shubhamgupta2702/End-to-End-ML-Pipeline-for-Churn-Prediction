import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score, r2_score
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import os 
import sys
from dataclasses import dataclass
from src.logger.logger import logger
from src.exception.exception import CustomException
from src.utils.utils import save_object, evaluate_model
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts', 'model.pkl')
  
class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
    
  def initiate_model_trainer(self, train_array, test_array):
    try:
      logger.info("Splitting training and testing data.")
      X_train, y_train, X_test, y_test = (
        train_array[:,:-1],
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1]
      )
      
      logger.info("Training different models")
      scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
      models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
        
        "Random Forest": RandomForestClassifier(random_state=42),
        
        "XGBoost Classifier": XGBClassifier(),
        
        "CatBoost Classifier": CatBoostClassifier(verbose=0, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost Classifier": AdaBoostClassifier()
      }
      
      params = {

    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "saga"],
        "max_iter": [500, 1000]
    },

    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    },

    "XGBoost Classifier": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 4, 6],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "gamma": [0, 0.1],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 1.5],
        "scale_pos_weight": [1, 3, 5]
    },

    "CatBoost Classifier": {
        "iterations": [100, 200],
        "learning_rate": [0.01, 0.05],
        "depth": [4, 6],
        "l2_leaf_reg": [3, 5, 7]
    },

    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 4],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "subsample": [0.8, 1.0]
    },

    "AdaBoost Classifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1]
    }
}
      
      model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params)
      
      print("\nModel Performance:")
      for model_name, score in model_report.items():
        print(f"{model_name}: {score:.4f}")
      
      best_model_score = max(sorted(model_report.values()))
      
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
      
      best_model = models[best_model_name]
      
      if best_model_score < 0.6:
        logger.info("No best model found")
        raise CustomException("No best model found")
        
      logger.info("Best model found on testing and training data.")
      
      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )
      logger.info("Best Model Dumped in artifacts")
      
      y_probs = best_model.predict_proba(X_test)[:, 1]
      
      print("\nThreshold Tuning:")
      
      preds = (y_probs >= 0.5).astype(int)
          
      precision = precision_score(y_test, preds)
      recall = recall_score(y_test, preds)
      f1 = f1_score(y_test, preds)
          
      print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
          
      # recall = recall_score(y_test, predicted)
      
      logger.info("prediction and recall_score is measured.")
      
      # precision = precision_score(y_test, predicted)
      # print("Precision:", precision)
      
      print(f'Best Model name: {models[best_model_name]}')
      logger.info(f"Model Trained Successfully.Best Model Name is : {{models[best_model_name]}}")
      
      return f1
      
    except Exception as e:
      logger.info("Error while dumping the model and training it.")
      raise CustomException(e, sys)
    