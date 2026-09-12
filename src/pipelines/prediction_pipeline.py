from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
import mlflow
import os

@dataclass
class PredictionPipelineConfig:
  modelUri: str = "runs:/9830f522484543dabf0f4ad6e699f92d/CatBoostModel"
  modelPath: str = os.path.join("artifacts", "model.joblib")
  featureEncoderObject: str = os.path.join("artifacts", "featureEncoderObject.joblib")
  kmeansModel: str = os.path.join("artifacts", "kmeansModel.joblib")

class PredictionPipeline:
  def __init__(self):
    self.predictionPipelineConfig = PredictionPipelineConfig()
  
  def predictResult(self, array: np.ndarray) -> int:
    # reshaping the array
    array = pd.DataFrame(array, index = ["job", "month", "balance", "duration", "campaign", "poutcome", "contacted_previously", "loans"]).T

    # encode features
    featureEncoder = joblib.load(self.predictionPipelineConfig.featureEncoderObject)
    array = featureEncoder.transform(array)

    # perform clustering
    kmeansModel = joblib.load(self.predictionPipelineConfig.kmeansModel)
    array = np.append(array, kmeansModel.predict(array).reshape((1, -1)))

    # loading the model
    if not os.path.isfile(self.predictionPipelineConfig.modelPath):
      modelName = "CatBoostModel"
      modelVersion = 28
      model = mlflow.pyfunc.load_model(self.predictionPipelineConfig.modelUri)
      joblib.dump(model, self.predictionPipelineConfig.modelPath)
    else:
      model = joblib.load(self.predictionPipelineConfig.modelPath)

    # inferencing the model
    result = model.predict(array)
    result = int(result)

    return result