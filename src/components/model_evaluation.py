from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from src.utils.exceptions import CustomException
from src.utils.logging import logger
from dataclasses import dataclass
from typing import Tuple
from dotenv import load_dotenv
import json
import os
import pandas as pd
import mlflow

load_dotenv("secrets.env")

@dataclass
class ModelEvaluationConfig:
    modelComparisonResults: str = os.path.join("artifacts", "results.json")
    dataPath: str = os.path.join("artifacts", "trainDataTransformed.csv")
    testDataPath: str = os.path.join("artifacts", "testDataTransformed.csv")

class ModelEvaluation:
    def __init__(self) -> None:
        logger.info(">>> MODEL EVALUATION STARTED <<<")
        self.modelEvaluationConfig = ModelEvaluationConfig()        

    def evaluateModel(self, params: dict) -> Tuple:
        try:
            logger.info("finding best model")
            results = json.load(open(self.modelEvaluationConfig.modelComparisonResults, "rb"))
            maxF1Score = max(results.values())
            for i in results.keys():
                if results[i] == maxF1Score:
                    bestModelKey = i
                else:
                    pass
            
            logger.info("configuring best model")
            models = {
                    "LogisticRegression": LogisticRegression,
                    "SVC": SVC,
                    "DecisionTreeClassifier": DecisionTreeClassifier,
                    "RandomForestClassifier": RandomForestClassifier,
                    "ExtraTreesClassifier": ExtraTreesClassifier,
                    "AdaBoostClassifier": AdaBoostClassifier,
                    "XGBClassifier": XGBClassifier,
                    "MLPClassifier": MLPClassifier,
                    "LGBMClassifier": LGBMClassifier,
                    "CatBoostClassifier": CatBoostClassifier
            }
            Model = models[bestModelKey]
            model = Model(**params)

            logger.info("fitting the model")
            data = pd.read_csv(self.modelEvaluationConfig.dataPath)
            test_data = pd.read_csv(self.modelEvaluationConfig.testDataPath)
            x_train, x_test, y_train, y_test = data.drop("y", axis = 1), test_data.drop("y", axis = 1), data["y"], test_data["y"]
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

            logger.info("evaluating the model")
            accuracy = accuracy_score(y_true = y_test, y_pred = model.predict(x_test))
            precision = precision_score(y_true = y_test, y_pred = model.predict(x_test))
            recall = recall_score(y_true = y_test, y_pred = model.predict(x_test))
            f1 = f1_score(y_true = y_test, y_pred = model.predict(x_test))
            roc_auc = roc_auc_score(y_true = y_test, y_score = model.predict_proba(x_test)[:, 1])

            return (accuracy, precision, recall, f1, roc_auc, model)

        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))
    
    def logExperiment(self) -> None:
        try:
            logger.info("configuring mlflow")
            mlflow.set_experiment("Bank-Marketing")

            logger.info("defining the parameters")
            params = {
                "iterations" : 750,
                "depth" : 5,
                "learning_rate" : 0.08,
                "class_weights" : {0 : 3, 1 : 7},
                "verbose" : False
            }

            logger.info("starting an MLflow run")
            with mlflow.start_run():
                # logging the hyperparameters
                mlflow.log_params(params)

                logger.info("logging the loss metric")
                accuracy, precision, recall, f1, roc_auc, model = self.evaluateModel(params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1 score", f1)
                mlflow.log_metric("roc-auc score", roc_auc)

                logger.info("logging the model")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="CatBoostModel",
                    registered_model_name="CatBoostModel",
                )
        
        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))