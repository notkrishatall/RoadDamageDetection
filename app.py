import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import io
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ AI-Based Road Damage Detection")
st.write("Upload a road image to detect road damage using our trained YOLOv5 model.")

MODEL_PATH = "best (4).pt"
@st.cache_resource
def load_model():
    return torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=MODEL_PATH,
        force_reload=False,
        trust_repo=True
    )
st.subheader("📍 Road Location")

location = streamlit_geolocation()

latitude = location.get("latitude") if location else None
longitude = location.get("longitude") if location else None

if latitude is not None and longitude is not None:
    st.success(f"Location captured: {latitude}, {longitude}")
    st.markdown(
        f"[📍 Open location in Google Maps]"
        f"(https://www.google.com/maps?q={latitude},{longitude})"
    )
else:
    st.info("Allow browser location permission to capture your location.")
image_source = st.radio(
    "Choose how to provide the road image:",
    ["📸 Take a Photo", " Upload an Image"],
    horizontal=True
)

if image_source == "📸 Take a Photo":
    uploaded_file = st.camera_input("Take a road photo")
else:
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
                    st.subheader("📊 Damage Summary")
                    st.write("D20:", int((detections["name"] == "D20").sum()))
                    st.write("D40:", int((detections["name"] == "D40").sum()))
                    d20_count = int((detections["name"] == "D20").sum())
                    d40_count = int((detections["name"] == "D40").sum())

                    voice_text = (
                        f"Road damage detection complete. "
                        f"Total {len(detections)} damaged areas detected. "
                        f"{d20_count} D20 damages and "
                        f"{d40_count} D40 damages detected."
                    )

                    import json
                    st.components.v1.html(
                        f"""
                        <button onclick='speakResult()'
                        style="padding:10px 16px;font-size:16px;cursor:pointer;">
                        🔊 Speak Results
                        </button>
                        <script>
                        function speakResult() {{
                            const msg = new SpeechSynthesisUtterance(
                                {json.dumps(voice_text)}
                            );
                            msg.lang = "en-US";
                            window.speechSynthesis.cancel();
                            window.speechSynthesis.speak(msg);
                        }}
                        </script>
                        """,
                        height=60
                    )

                    if latitude is not None and longitude is not None:
                        st.markdown(
                            f"[📍 Open Google Maps](https://www.google.com/maps?q={latitude},{longitude})"
                        )
                else:
                    st.info("No damage detected above the confidence threshold.")
# Save Detection Report
st.subheader("📋 Detection Report")

if uploaded_file and "detections" in locals():
    if len(detections) > 0:
        report = detections.copy()
        report["date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report["image_name"] = uploaded_file.name
        report["latitude"] = latitude
        report["longitude"] = longitude
        csv_data = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Detection Report (CSV)",
            data=csv_data,
            file_name="road_damage_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No road damage detected in this image.")
