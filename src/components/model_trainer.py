import sys 
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("Train test split done")

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regressor": LinearRegression(),
                "Decision tree": DecisionTreeRegressor(),
                "XgBoost": XGBRegressor(),
                "Gradient Boost Regressor": GradientBoostingRegressor(),
                "ADA Boost Regressor": AdaBoostRegressor(),
                "K Neighbours":KNeighborsRegressor(),
                "Cat Boost Regressor": CatBoostRegressor()
                }
            
            logging.info("Model training initiated")
            
            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models=models)

            logging.info(f"Model training and evaluation completed{model_report}")

            best_model_score = max(model_report.values())

            logging.info("Best model score: {}".format(best_model_score))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info(f"Best model found on both training and testing data: {best_model}")


            if best_model_score<= 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found on both training and testing data")

            save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
            )            
            logging.info("Best model saved in the pickle file")

            predicted = best_model.predict(X_test)
            logging.info("Predicted on test data")
            r2_score_value = r2_score(y_test, predicted)

            return r2_score_value

        except Exception as e:
            CustomException(e,sys)
            