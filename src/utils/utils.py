import os
import sys
import joblib
import dill
from src.logger.logger import logger
from src.exception.exception import CustomException
from src.exception.exception import CustomException
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, "wb") as file_obj:
      dill.dump(obj, file_obj)
      logger.info("Succesfully executed save_object in utils.py")
      
    
      
  except Exception as e:
    logger.error("Error in save_object function in utils.py")
    raise CustomException(e, sys)
  
  
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
  try:
    report = {}
    for i in range(len(list(models))):
      model = list(models.values())[i]
      param_grid = param[list(models.keys())[i]]
      
      grid = GridSearchCV(
      estimator=model,
      param_grid=param_grid,
      cv=3,
      scoring="f1",
      n_jobs=-1,
      verbose=2
      )
      
      grid.fit(X_train, y_train)
      
      model.set_params(**grid.best_params_)
      model.fit(X_train,y_train)
      
      y_train_pred = model.predict(X_train)
      y_test_pred = model.predict(X_test)
      
      train_model_score = recall_score(y_train, y_train_pred)
      test_model_score = f1_score(y_test, y_test_pred)
      
      report[list(models.keys())[i]] = test_model_score
      logger.info("Evaluate Model function executed in utils.py")
      
    return report

  except Exception as e:
    logger.info("Error in evaluate model function in utils.py")
    raise CustomException(e, sys)
  
  
def load_object(file_path:str):
  try:
    with open(file_path, "rb") as file_obj:
      model = dill.load(file_obj)
      logger.info("Succesfully loaded the object in utils.py")
      return model
      
  except Exception as e:
    logger.info("Error in load_object function in utils.py")
    raise CustomException(e, sys)
    
  