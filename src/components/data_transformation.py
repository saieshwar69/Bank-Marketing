from src.utils.logging import logger
from src.utils.exceptions import CustomException
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import joblib
from typing import Tuple
import os

@dataclass
class DataTransformationConfig:
    trainDataPath: str = os.path.join("artifacts", "trainData.csv")
    testDataPath: str = os.path.join("artifacts", "testData.csv")
    featureEncoderPath: str = os.path.join("artifacts", "featureEncoderObject.joblib")
    missingValuesImputerPath: str = os.path.join("artifacts", "missingValuesImputerObject.joblib")
    kmeansModelPath: str = os.path.join("artifacts", "kmeansModel.joblib")
    trainDataTransformed: str = os.path.join("artifacts", "trainDataTransformed.csv")
    testDataTransformed: str = os.path.join("artifacts", "testDataTransformed.csv")

class DataTransformation:
    def __init__(self) -> None:
        logger.info(">>> DATA TRANSFORMATION STARTED <<<")
        self.dataTransformationConfig = DataTransformationConfig()

    def basicFeatureEngineering(self) -> Tuple[pd.DataFrame]:
        try:
            # loading the csv data
            data = pd.read_csv(self.dataTransformationConfig.trainDataPath)
            test_data = pd.read_csv(self.dataTransformationConfig.testDataPath)

            logger.info("checking for missing values")
            for col in data.columns:
                if data[col].dtype == "object":
                    data[col].replace(to_replace = "unknown", value = np.NaN, inplace = True)
                else:
                    pass
            for col in test_data.columns:
                if test_data[col].dtype == "object":
                    test_data[col].replace(to_replace = "unknown", value = np.NaN, inplace = True)
                else:
                    pass

            logger.info("constructing features")
            data["contacted_previously"] = data["pdays"].apply(lambda x : 0 if x == -1 else x)
            test_data["contacted_previously"] = test_data["pdays"].apply(lambda x : 0 if x == -1 else x)
            data["loans"] = data["housing"].apply(lambda x : 1 if x == "yes" else 0) + data["loan"].apply(lambda x : 1 if x == "yes" else 0)
            test_data["loans"] = test_data["housing"].apply(lambda x : 1 if x == "yes" else 0) + test_data["loan"].apply(lambda x : 1 if x == "yes" else 0)

            logger.info("performing feature selection")
            features_to_be_dropped = ["marital", "education", "default", "contact", "day", "age", "pdays", "previous", "housing", "loan"]
            data = data.drop(features_to_be_dropped, axis = 1)
            test_data = test_data.drop(features_to_be_dropped, axis = 1)

            # modifying "poutcome"
            data["poutcome"] = data["poutcome"].apply(lambda x : 1 if x == "success" else 0)
            test_data["poutcome"] = test_data["poutcome"].apply(lambda x : 1 if x == "success" else 0)

            logger.info("performing outlier removal")
            for col in ["duration", "campaign", "balance"]:
                q3 = np.percentile(data[col], 75)
                q1 = np.percentile(data[col], 25)
                iqr = q3 - q1
                upper_bound = q3 + 1.75 * iqr
                lower_bound = q3 - 1.75 * iqr
                data = data[(data[col] <= upper_bound) & (data[col] >= lower_bound)]

            return (data, test_data)
        
        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))

    def featureEncoding(self, data, test_data) -> Tuple[pd.DataFrame]:
        try:
            logger.info("encoding target variable")
            data["y"].replace(to_replace = ["no", "yes"], value = [0, 1], inplace = True)
            test_data["y"].replace(to_replace = ["no", "yes"], value = [0, 1], inplace = True)

            logger.info("performing label encoding on categoricals")
            x_train, y_train, x_test, y_test = data.drop("y", axis = 1), data["y"], test_data.drop("y", axis = 1), test_data["y"] 

            transformer = ColumnTransformer(
                transformers=[("categoricalFeature", OrdinalEncoder(), [x for x in x_train.columns if x_train[x].dtype == "O"])],
                remainder = "passthrough"
            )

            x_train = pd.DataFrame(transformer.fit_transform(x_train), columns = [x.split("__")[1] for x in transformer.get_feature_names_out()], index = y_train.index)
            x_test = pd.DataFrame(transformer.transform(x_test), columns = [x.split("__")[1] for x in transformer.get_feature_names_out()], index = y_test.index)

            data = pd.concat([x_train, y_train], axis = 1)
            test_data = pd.concat([x_test, y_test], axis = 1)

            logger.info("saving the encoder object")
            joblib.dump(transformer, self.dataTransformationConfig.featureEncoderPath)

            return (data, test_data)
    
        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))

    def missingValuesImputation(self, data, test_data) -> Tuple[pd.DataFrame]:
        try:
            logger.info("imputing missing values")
            imputer = SimpleImputer(strategy = "most_frequent")
            data = pd.DataFrame(imputer.fit_transform(data), columns = imputer.feature_names_in_)
            test_data = pd.DataFrame(imputer.transform(test_data), columns = imputer.feature_names_in_)
            
            logger.info("saving the imputer object")
            joblib.dump(imputer, self.dataTransformationConfig.missingValuesImputerPath)
            return (data, test_data)
        
        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))

    def clustering(self, data, test_data) -> None:
        try:
            logger.info("performing k-means clustering")
            kmeans = KMeans(n_clusters = 3, init = "k-means++")
            data["cluster"] = kmeans.fit_predict(data.drop("y", axis = 1), data["y"])
            test_data["cluster"] = kmeans.predict(test_data.drop("y", axis = 1), test_data["y"])

            logger.info("saving the k-means model")
            joblib.dump(kmeans, self.dataTransformationConfig.kmeansModelPath)

            data.to_csv(self.dataTransformationConfig.trainDataTransformed, index = False)
            test_data.to_csv(self.dataTransformationConfig.testDataTransformed, index = False)

        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))