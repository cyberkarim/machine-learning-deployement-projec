from fastapi import FastAPI
import univcorn
from pydantic import BaseModel
import pickle
import os
from pathlib import Path
import json
import csv

class upload_image(BaseModel):
    uploaded_data : json

class feedback(BaseModel):
    uploaded_data: json
    prediction: str
    target: str

def get_project_root() -> Path:
    return Path(__file__).parent.parent

# Function to unplickle artifact files
def unpickle_embedding_model(path):    
    with open(path, 'rb') as fo:
        embedding = pickle.load(fo, encoding='bytes')
    return embedding

def unpickle_scaler_model(path):    
    with open(path, 'rb') as fo:
        scaler = pickle.load(fo, encoding='bytes')
    return scaler

def unpickle_prediction_model(path):    
    with open(path, 'rb') as fo:
        predictor = pickle.load(fo, encoding='bytes')
    return predictor


if __name__=="__main__":
    # Get path of the project
    project_root = get_project_root()

    # Get path of artifact files
    scaler_path = os.path.join(project_root,"artifact/scaler.pkl")
    predictor_path = os.path.join(project_root,"artifact/predictor.pkl")
    embedding_path = os.path.join(project_root,"artifact/embedding.pkl")

    # Unpickle artifact files
    scaler_model = unpickle_scaler_model(scaler_path)
    predictor_model = unpickle_prediction_model(predictor_path)
    embedding_model=unpickle_embedding_model(embedding_path)
    
    # Create fastAPI api
    app = FastAPI()
    # "predict" endpoint
    @app.post("/predict/")
    async def generate_anomaly_score(Data: upload_image):
        uploaded_image = Data.uploaded_data
        scaled_data =  scaler_model.fit(uploaded_image)
        prediction = predictor_model.fit(scaled_data)

        return prediction
    
    # "feedback" endpoint
    @app.post("/feedback/")
    async def feedback_image(Data: feedback):
        # get path of prod_data.csv
        prod_data_path=get_project_root+"/data/prod_data.csv"

        # fit uploaded image to insert into prod_data.csv
        upload_image=feedback.uploaded_data
        prediction=feedback.prediction
        target=feedback.target

        upload_image_embedding=embedding_model(upload_image)
        new_data=upload_image_embedding+str(prediction)+str(target)
        with open(prod_data_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(new_data)

        