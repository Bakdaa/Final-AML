import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load pre-trained model and components
@st.cache_resource
def load_components():
    model = load_model("custom_model.h5")
    scaler = StandardScaler()
    pca = PCA(n_components=561)
    # Assume scaler and PCA were fitted during training
    return model, scaler, pca

model, scaler, pca = load_components()

activity_labels = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

def preprocess_frame(frame, scaler, pca):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (23, 25))
    flat_frame = resized_frame.flatten()[:561]
    flat_frame = flat_frame / 255.0
    flat_frame = flat_frame.reshape(1, -1)
    scaled_frame = scaler.transform(flat_frame)
    reduced_frame = pca.transform(scaled_frame)
    return reduced_frame

st.title("Activity Recognition Demo")
st.write("Upload a video frame or capture live activity using your webcam.")

# File Upload
uploaded_file = st.file_uploader("Upload a video frame (image file)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(frame, caption="Uploaded Frame", use_column_width=True)

    # Preprocess and predict
    preprocessed = preprocess_frame(frame, scaler, pca)
    prediction = model.predict(preprocessed)
    activity = activity_labels[np.argmax(prediction)]
    st.write(f"Predicted Activity: **{activity}**")

# Webcam Capture
if st.button("Capture from Webcam"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible!")
    else:
        ret, frame = cap.read()
        if ret:
            st.image(frame, caption="Captured Frame", use_column_width=True)
            preprocessed = preprocess_frame(frame, scaler, pca)
            prediction = model.predict(preprocessed)
            activity = activity_labels[np.argmax(prediction)]
            st.write(f"Predicted Activity: **{activity}**")
        cap.release()
