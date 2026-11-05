from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "waste_classifier.keras"
LABEL_PATH = ROOT / "models" / "class_names.json"

GUIDANCE = {
    "cardboard": ("♻️ Usually recyclable", "Keep it dry and clean. Flatten boxes and remove plastic tape where practical."),
    "glass": ("♻️ Often recyclable", "Separate glass carefully. Do not handle broken glass with bare hands; follow local collection rules."),
    "metal": ("♻️ Usually recyclable", "Empty food or drink cans and rinse if required by your local recycler."),
    "paper": ("♻️ Usually recyclable", "Keep paper clean and dry. Greasy or food-soiled paper may not be accepted."),
    "plastic": ("♻️ Depends on resin and local rules", "Empty and rinse containers. Check the recycling symbol and local collection rules; not all plastics are accepted."),
    "trash": ("🗑️ General waste category", "This model's 'trash' class is a broad dataset label. Check local rules for safe disposal; do not assume every item is non-recyclable.")
}

st.set_page_config(page_title="EcoSort AI | Waste Classifier", page_icon="♻️", layout="centered")
st.title("♻️ EcoSort AI")
st.subheader("Smart Waste Classification & Recycling Assistant")
st.write("Upload a waste image or capture one with your camera. The trained ML model predicts one of six categories.")

if not MODEL_PATH.exists() or not LABEL_PATH.exists():
    st.warning("Model not found. First run the setup, download the dataset, prepare it and train the model. See README.md.")
    st.code("python download_dataset.py\npython prepare_data.py\npython train.py\nstreamlit run app.py")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = json.loads(LABEL_PATH.read_text())

upload = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png"])
camera = st.camera_input("Or take a photo")

source = upload if upload is not None else camera
if source is not None:
    image = Image.open(source).convert("RGB")
    st.image(image, caption="Selected image", use_container_width=True)
    resized = image.resize((224, 224))
    arr = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    best_idx = int(order[0])
    label = class_names[best_idx]
    confidence = float(probabilities[best_idx])

    st.markdown("---")
    st.metric("Predicted waste category", label.title(), f"{confidence*100:.1f}% confidence")
    if confidence < 0.55:
        st.warning("The model is not very confident. Try a clearer photo with one item, good lighting and less background clutter.")
    else:
        st.success("Prediction generated successfully.")

    title, advice = GUIDANCE.get(label, ("Check local rules", "Follow local waste-management guidance."))
    st.markdown(f"### {title}")
    st.write(advice)

    st.markdown("### Top 3 predictions")
    for idx in order[:3]:
        st.write(f"**{class_names[int(idx)].title()}** — {probabilities[idx]*100:.2f}%")
        st.progress(float(probabilities[idx]))

st.markdown("---")
with st.expander("About this project"):
    st.write("EcoSort AI uses MobileNetV2 transfer learning, a convolutional neural network architecture, trained on the TrashNet six-class dataset.")
    st.caption("Educational prototype only. Predictions may be wrong, especially for mixed, dirty, unusual or poorly lit waste. Follow local municipal recycling rules.")
