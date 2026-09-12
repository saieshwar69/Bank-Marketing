from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation 

if __name__ == "__main__":
    # data ingestion
    dataIngestionObj = DataIngestion()
    dataIngestionObj.extractData()
    dataIngestionObj.splitData()

    # data transformation
    dataTransformationObj = DataTransformation()
    trainData, testData = dataTransformationObj.basicFeatureEngineering()
    trainData, testData = dataTransformationObj.featureEncoding(trainData, testData)
    trainData, testData = dataTransformationObj.missingValuesImputation(trainData, testData)
    dataTransformationObj.clustering(trainData, testData)

    # model selection and training
    modelTrainer = ModelTrainer()
    modelTrainer.selectModel()

    # model evaluation with MLFlow
    modelEvaluationObj = ModelEvaluation()
    modelEvaluationObj.logExperiment()