import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import io
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components

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
        "ultralytics/yolov5",
        "custom",
        path=MODEL_PATH,
        force_reload=False,
        trust_repo=True
    )
st.subheader("📍 Road Location")
latitude = st.number_input("Latitude", format="%.6f", value=0.0)
longitude = st.number_input("Longitude", format="%.6f", value=0.0)
components.html("""
<div>
  <button onclick="getLocation()">Get My Live Location</button>
  <p id="status">“Click the button to get your location.”.</p>
  <script>
    function getLocation() {
      const status = document.getElementById("status");
      if (!navigator.geolocation) {
        status.innerText = "Browser location support nahi karta.";
        return;
      }
      status.innerText = "Location permission ka wait ho raha hai...";
      navigator.geolocation.getCurrentPosition(
        function(position) {
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;
          status.innerHTML =
            "Latitude: " + lat + "<br>Longitude: " + lon +
            '<br><a target="_blank" href="https://www.google.com/maps?q='
            + lat + ',' + lon + '">Open in Google Maps</a>';
        },
        function(error) {
          status.innerText = "Location nahi mili. Browser permission allow karo.";
        },
        {enableHighAccuracy: true, timeout: 15000}
      );
    }
  </script>
</div>
""", height=150)
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
# Save Detection Report
st.subheader("📋 Detection Report")

if uploaded_file and "detections" in locals():
    if len(detections) > 0:
        report = detections.copy()
        report["date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report["image_name"] = uploaded_file.name
report["latitude"] = latitude if latitude != 0 else None
report["longitude"] = longitude if longitude != 0 else None
        csv_data = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Detection Report (CSV)",
            data=csv_data,
            file_name="road_damage_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No road damage detected in this image.")
