from PIL import Image
import streamlit as st
import numpy as np
import requests
import json
import os


webapp_path=os.path.dirname(os.path.abspath(__file__))

st.title("Interface web streamlite")
st.write("Projet ML1 - Karouma Youssef _ Khatib Mohamed _ Majdi Karim _ Tibi Daniel")
uploaded_image = st.file_uploader("Téléchargez une image", type=["png"])

if uploaded_image is not None:

    uploaded_image_save_path=webapp_path+"/test/"+uploaded_image.name
    with open(uploaded_image_save_path, "wb") as f:
        f.write(uploaded_image.read())


    uploaded_image={
        "link":uploaded_image_save_path
    }
    response = requests.post(url="http://127.0.0.1:8000/predict",json=uploaded_image)

    if response.status_code == 200:
        prediction=response.json()
        # recuperer en format json la requete post
        st.write("Sucess request")
        st.session_state["prediction"]=prediction
        st.write(prediction)


target = st.text_area("Entrer le vrai label:",max_chars=10)

if st.button("Soumettre"):
    # Afficher la réponse
    feedback_image={
        "link": uploaded_image_save_path,
        "prediction": prediction,
        "target": target
    }
    
    response=requests.post("http://127.0.0.1:8000/feedback", json=feedback_image)

    if response.json()==1:
        st.write("Sucess feedback")


    