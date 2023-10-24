import streamlit as st
import requests

st.title("Interface web streamlite")
st.write("Projet ML1 - Karouma Youssef _ Khatib Mohamed _ Majdi Karim _ Tibi Daniel")
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])
if st.button("Predire"):
    api_url = " http://serving-api:8080/predict"
    response = requests.get(api_url)

    if response.status_code == 200:
        st.success("Appel API réussi !")
    else:
        st.error("Échec de l'appel API")
