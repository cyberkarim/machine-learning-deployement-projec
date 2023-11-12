import streamlit as st
from pydantic import BaseModel
import requests
import json

class upload_image(BaseModel):
    uploaded_data : json

class feedback(BaseModel):
    uploaded_data: json
    prediction: str
    target: str

st.title("Interface web streamlite")
st.write("Projet ML1 - Karouma Youssef _ Khatib Mohamed _ Majdi Karim _ Tibi Daniel")
uploaded_image = st.file_uploader("Téléchargez une image", type=["png"])


if uploaded_image is not None:
    # que mettre dans data_link
    uploaded_image=json.loads(uploaded_image.decode('utf-8'))
    uploaded_image=upload_image(uploaded_data=uploaded_image)
    response = requests.post("http://serving-api:8080/predict", Data=uploaded_image)

    if response.status_code == 200:
        prediction=response.json()
        # recuperer en format json la requete post
        st.write("Sucess request")
        st.write(prediction)
    else:
        st.write("Denied request")

    if st.button("Send Feedback"):
        feedback_text = st.text_area("Rentrer le vrai label de l'image", max_chars=200)
        feedback_image=feedback(uploaded_data=upload_image,prediction=prediction,target=feedback_text)

        # améliorer le modele avec un feedback utilisateur 
        requests.post("http://serving-api:8080/feedback", Data=feedback_image)

