import streamlit as st
import requests

st.title("Interface web streamlite")
st.write("Projet ML1 - Karouma Youssef _ Khatib Mohamed _ Majdi Karim _ Tibi Daniel")
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    files = {'file': uploaded_file}
    response = requests.post("http://localhost:8080", files=files)

    if response.status_code == 200:
        st.write("Sucess request")
    else:
        st.write("Denied request")
