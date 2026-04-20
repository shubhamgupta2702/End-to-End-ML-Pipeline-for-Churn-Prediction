import os
import sys
import dill
from src.logger.logger import logger
from src.exception.exception import CustomException


def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, "wb") as file_obj:
      dill.dump(obj, file_obj)
      logger.info("Succesfully executed save_object in utils.py and dumped preprocessing.pkl")
      
    
      
  except Exception as e:
    logger.error("Error in save_object function in utils.py")
    raise CustomException(e, sys)