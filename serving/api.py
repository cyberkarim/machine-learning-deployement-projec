from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
def get_project_root() -> Path:
    return Path(__file__).parent.parent

def unpickle_embedding_model():
    
    with open(, 'rb') as fo:
        scaler = pickle.load(fo, encoding='bytes')
    return scaler

def unpickle_scaler_model(path):
    
    with open(path, 'rb') as fo:
        scaler = pickle.load(fo, encoding='bytes')
    return scaler

def unpickle_prediction_model(path):
    
    with open(path, 'rb') as fo:
        predictor = pickle.load(fo, encoding='bytes')
    return predictor

class Data_point(BaseModel):
    data_link : str

##{
##data_numerical_vector : []
  
##}    

if __name__=="__main__":
  project_root = get_project_root()
  scaler_path = os.path.join(project_root,"artifact/scaler.pkl")
  predictor_path = os.path.join(project_root,"artifact/predictor.pkl")
  scaler_model = unpickle_scaler_model(scaler_path)
  predictor_model = unpickle_scaler_model(predictor_path)

  app = FastAPI()
  @app.post("/predict/")
  async def generate_anomaly_score(Data: Data_point):
    inference_data_point = Data.data_link

    scaled_data =  scaler_model.fit(inference_data_point)

    anomaly_score = predictor_model.fit(inference_data_point)

    return anomaly_score
