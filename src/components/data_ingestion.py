import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.utils.logging import logger
from src.utils.exceptions import CustomException

load_dotenv("secrets.env")

@dataclass
class DataIngestionConfig:
    artifactsDir: str = os.path.join(os.getcwd(), "artifacts")
    dataPath: str = os.path.join("artifacts", "data.csv")
    trainDataPath: str = os.path.join("artifacts", "trainData.csv")
    testDataPath: str = os.path.join("artifacts", "testData.csv")

class DataIngestion:
    def __init__(self) -> None:
        logger.info(">>> DATA INGESTION STARTED <<<")
        self.dataIngestionConfig = DataIngestionConfig()
        os.system("curl --create-dirs -o $HOME/.postgresql/root.crt 'https://cockroachlabs.cloud/clusters/e5210205-9ccd-49c3-ae85-dae85bf87ac0/cert'")

    def extractData(self) -> None:
        try:
            logger.info("executing loadData function")
            password = os.getenv("DB_PASSWORD")
            connection_string = f"postgresql://rauhan:{password}@bank-marketing-4019.7s5.aws-ap-south-1.cockroachlabs.cloud:26257/BankMarketing?sslmode=verify-full"
            conn = psycopg2.connect(connection_string)

            logger.info("extracting data from the database")
            with conn.cursor() as cur:
                cur.execute('SELECT * FROM public."bank-full";')
                colnames = [desc[0] for desc in cur.description]
                res = cur.fetchall()
                conn.commit()
            data = pd.DataFrame(res, columns = colnames)
            
            logger.info("saving data to artifacts")
            os.makedirs(self.dataIngestionConfig.artifactsDir, exist_ok = True)
            data.to_csv(self.dataIngestionConfig.dataPath, index = False)
        
        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))

    def splitData(self) -> None:
        try:
            data = pd.read_csv(self.dataIngestionConfig.dataPath)

            logger.info("splitting the dataset")
            X, y = data.drop("y", axis = 1), data["y"]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
            data = pd.concat([x_train, y_train], axis = 1)
            testData = pd.concat([x_test, y_test], axis = 1)

            logger.info("saving the train and test data")
            data.to_csv(self.dataIngestionConfig.trainDataPath, index = False)
            testData.to_csv(self.dataIngestionConfig.testDataPath, index = False)
        
        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))