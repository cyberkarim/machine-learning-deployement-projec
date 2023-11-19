from fastapi import Body, FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from PIL import Image
import json
import csv
import cv2



class Link_upload_image(BaseModel):
    link= str

class feedback(BaseModel):
    uploaded_data= dict
    prediction= str
    target= str

def get_project_root() -> Path:
    return Path(__file__).parent.parent

# Function to unplickle model in artifact file 
def unpickle_model(path):    
    with open(path, 'rb') as fo:
        model = pickle.load(fo)
    return model

# Import embedding and scaler
def embedding(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(8,8))
    img=img.reshape(-1) # flatten the matrix
    # print(type(img))
    return img

def normalize_image(array):
    return array/255

#project_root=get_project_root()

prod_data="data/prod_data.csv"
model_path = "artifacts/model.pkl"
model=unpickle_model(model_path)

# Create fastAPI api
app = FastAPI()
# "predict" endpoint
@app.post("/predict")
def predict_image(upload_image: dict=Body(...)):

    upload_image=Image.open(upload_image["link"])
    upload_image=np.array(upload_image)

    upload_image=embedding(upload_image)
    upload_image=normalize_image(upload_image)

    prediction=model[0].predict(upload_image.reshape(1,-1))[0]
    return prediction

    
# "feedback" endpoint
@app.post("/feedback")
def save_feedback(feedback: dict=Body(...)):
    
    # fit uploaded image to insert into prod_data.csv
    upload_image=feedback["link"]
    prediction=feedback["prediction"]
    target=feedback["target"]

    upload_image=Image.open(upload_image)
    upload_image=np.array(upload_image)

    upload_image=embedding(upload_image)

    upload_image=list(upload_image)

    upload_image.append(int(prediction))
    upload_image.append(int(target))

    upload_image=map(str,upload_image)

    with open(prod_data, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter=";")
        csv_writer.writerow(upload_image)

    return 1
