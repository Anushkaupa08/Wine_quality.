import streamlit as st
import tensorflow as tf
import numpy as np

# Page setting
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #8B0000;'>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.write("Check the quality of Red Wine from 3 to 8 based on 11 chemical properties")
st.divider()

# Model load - upar le aaye taaki bar load na ho
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('wine_quality_model.h5')

model = load_model()

# 2 Column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Acidity & Sugar")
    fixed_acidity = st.number_input("Fixed Acidity", min_value=4.0, max_value=16.0, value=7.4, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.1, max_value=1.5, value=0.7, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.9, max_value=15.0, value=1.9, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.01, max_value=0.6, value=0.08, step=0.001)
    pH = st.number_input("pH", min_value=2.5, max_value=4.5, value=3.51, step=0.01)

with col2:
    st.subheader("Other Properties")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=1, max_value=72, value=11, step=1)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=6, max_value=289, value=34, step=1)
    density = st.number_input("Density", min_value=0.990, max_value=1.005, value=0.9978, step=0.0001)
    sulphates = st.number_input("Sulphates", min_value=0.3, max_value=2.0, value=0.56, step=0.01)
    alcohol = st.number_input("Alcohol %", min_value=8.0, max_value=15.0, value=9.4, step=0.1)

st.divider()

# Predict button
if st.button("🔮 Predict Quality", use_container_width=True, type="primary"):
    # FIX 1: Order sahi kiya - pH ko 4th position pe
    input_data = [[fixed_acidity, volatile_acidity, citric_acid, pH, residual_sugar, 
                   chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, sulphates, alcohol]]
    
    prediction = model.predict(np.array(input_data))
    
    # FIX 2: +3 bracket ke bahar
    pred = np.argmax(prediction) + 3
    
    # Result ko colorful box me
    st.success(f"### Predicted Wine Quality: {pred} / 10")
    
    if pred >= 7:
        st.balloons()
        st.info("✨ Premium Quality Wine!")
    elif pred >= 5:
        st.warning("👍 Average Quality Wine")
    else:
        st.error("😞 Low Quality Wine")

st.caption("Made with ❤️ using Streamlit + TensorFlow")