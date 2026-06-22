import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import os
import pickle
from PIL import Image
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

st.title("Reconnaissance Faciale 🎯")
st.write("Charge une photo pour identifier l'acteur ou l'actrice !")

@st.cache_resource
def load_models():
    device = torch.device("cpu")
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    classifier = None
    le = None
    if os.path.exists("models/face_classifier.pkl") and os.path.exists("models/label_encoder.pkl"):
        with open("models/face_classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    return mtcnn, resnet, classifier, le, device

mtcnn, resnet, classifier, le, device = load_models()

uploaded_file = st.file_uploader("Choisis une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # LA CORRECTION EST ICI : on force l'image à abandonner sa transparence (.convert('RGB'))
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image chargée", use_container_width=True)
    
    if classifier is None or le is None:
        st.error("Les modèles d'entraînement sont introuvables.")
    else:
        st.write("Analyse de l'image en cours...")
        img_cropped_list, prob_list = mtcnn(image, return_prob=True)
        
        if img_cropped_list is not None:
            for i, img in enumerate(img_cropped_list):
                if prob_list[i] > 0.90:
                    emb = resnet(img.unsqueeze(0).to(device)).detach().numpy()
                    prediction = classifier.predict(emb)
                    name = le.inverse_transform(prediction)[0]
                    st.success(f"Visage détecté : **{name}** ✨")
                else:
                    st.warning("Visage détecté mais douteux.")
        else:
            st.error("Aucun visage détecté sur cette photo.")

