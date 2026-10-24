import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import io

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ AI-Based Road Damage Detection")
st.write("Upload a road image to detect road damage using our trained YOLOv5 model.")

MODEL_PATH = "best (2).pt"
@st.cache_resource
def load_model():
    return torch.hub.load(
        "/content/yolov5",
        "custom",
        path=MODEL_PATH,
        source="local"
    )

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    st.image(image, caption="Uploaded road image", use_container_width=True)

    if st.button("Detect Road Damage", type="primary"):
        if not Path(MODEL_PATH).exists():
            st.error("Model file not found. Check Google Drive mounting.")
        else:
            with st.spinner("Analyzing image..."):
                model = load_model()
                results = model(image)
                detected_image = results.render()[0][:, :, ::-1]
                st.subheader("Detection Results")
                st.image(detected_image, caption="AI detection", use_container_width=True)

                detections = results.pandas().xyxy[0]
                st.write("**Detected objects:**", len(detections))

                if len(detections):
                    st.dataframe(
                        detections[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]],
                        use_container_width=True
                    )
                else:
                    st.info("No damage detected above the confidence threshold.")
