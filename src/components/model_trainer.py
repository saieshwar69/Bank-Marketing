from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from src.utils.logging import logger
from src.utils.exceptions import CustomException
import pandas as pd
import json
import os
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    dataPath: str = os.path.join("artifacts", "trainDataTransformed.csv")
    resultsPath: str = os.path.join("artifacts", "results.json")

class ModelTrainer:
    def __init__(self) -> None:
        logger.info(">>> MODEL SELECTION STARTED <<<")
        self.modelTrainerConfig = ModelTrainerConfig()

    def selectModel(self) -> None:
        try:
            data = pd.read_csv(self.modelTrainerConfig.dataPath)
            
            logger.info("training different models")
            models = [
                LogisticRegression(),
                SVC(),
                DecisionTreeClassifier(),
                RandomForestClassifier(),
                ExtraTreesClassifier(),
                AdaBoostClassifier(),
                XGBClassifier(),
                MLPClassifier(),
                LGBMClassifier(),
                CatBoostClassifier()
            ]

            x_train, x_test, y_train, y_test = train_test_split(data.drop("y", axis = 1), data["y"], test_size = 0.2)

            logger.info("saving their f1-scores")
            f1_scores = []
            for model in models:
                model.fit(x_train, y_train)
                score = f1_score(y_true = y_test, y_pred = model.predict(x_test))
                f1_scores.append(score)
        
            model_names = [
                "LogisticRegression",
                "SVC",
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "ExtraTreesClassifier",
                "AdaBoostClassifier",
                "XGBClassifier",
                "MLPClassifier",
                "LGBMClassifier",
                "CatBoostClassifier"
            ]

            logger.info("storing the results in a dictionary")
            results = {}
            for modelName, f1Score in zip(model_names, f1_scores):
                results[modelName] = f1Score
            
            logger.info("writing out the results as a json file")
            with open(self.modelTrainerConfig.resultsPath, "w") as file:
                json.dump(results, file)

        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))