import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import io
import pandas as pd
import json
from datetime import datetime
import streamlit.components.v1 as components
from streamlit_geolocation import streamlit_geolocation
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ AI-Based Road Damage Detection")
st.write(
    "Upload a road image or take a photo to detect road damage "
    "using our trained YOLOv5 model."
)

MODEL_PATH = "best (4).pt"
def get_image_gps(image):
    try:
        exif = image.getexif()
        gps_info = None

        for tag_id, value in exif.items():
            if TAGS.get(tag_id) == "GPSInfo":
                gps_info = value
                break

        if not gps_info:
            return None, None

        gps = {
            GPSTAGS.get(key, key): value
            for key, value in gps_info.items()
        }

        def to_degrees(value):
            return (
                float(value[0])
                + float(value[1]) / 60
                + float(value[2]) / 3600
            )

        lat = to_degrees(gps["GPSLatitude"])
        lon = to_degrees(gps["GPSLongitude"])

        if gps.get("GPSLatitudeRef") == "S":
            lat = -lat

        if gps.get("GPSLongitudeRef") == "W":
            lon = -lon

        return lat, lon

    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None, None


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=MODEL_PATH,
        force_reload=False,
        trust_repo=True
    )


# ---------------- GPS LOCATION ----------------
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
    st.info(
        "Allow browser location permission to capture your current location."
    )


# ---------------- IMAGE INPUT ----------------
image_source = st.radio(
    "Choose how to provide the road image:",
    ["📸 Take a Photo", "📁 Upload an Image"],
    index=1
)

if image_source == "📸 Take a Photo":
    uploaded_file = st.camera_input("Take a road photo")
else:
    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"]
    )


# ---------------- DETECTION ----------------
if uploaded_file is not None:

    original_image = Image.open(
    io.BytesIO(uploaded_file.getvalue())
)

# Read GPS from the original image before RGB conversion
photo_latitude, photo_longitude = get_image_gps(original_image)

# Convert image for YOLOv5 detection
image = original_image.convert("RGB")

if photo_latitude is not None and photo_longitude is not None:
    st.success(
        f"📍 Photo GPS: {photo_latitude:.6f}, {photo_longitude:.6f}"
    )
    st.markdown(
        f"[Open photo location in Google Maps]"
        f"(https://www.google.com/maps?q={photo_latitude},{photo_longitude})"
    )
else:
    st.info(
        "No GPS metadata found in this photo. "
        "The current device location will be used if available."
    )

    st.image(
        image,
        caption="Selected road image",
        use_container_width=True
    )

    if st.button("Detect Road Damage", type="primary"):

        if not Path(MODEL_PATH).exists():
            st.error(
                f"Model file not found: {MODEL_PATH}. "
                "Make sure the model file is available in your project."
            )

        else:
            try:
                with st.spinner("Analyzing image..."):

                    model = load_model()
                    results = model(image)

                    # Render detection image
                    detected_image = results.render()[0][:, :, ::-1]

                    st.subheader("🔍 Detection Results")
                    st.image(
                        detected_image,
                        caption="AI Detection",
                        use_container_width=True
                    )

                    # Detection data
                    detections = results.pandas().xyxy[0].copy()

                    st.write("**Detected objects:**", len(detections))

                    # ---------------- DAMAGE SUMMARY ----------------
                    st.subheader("📊 Damage Summary")

                    d20_count = int(
                        (detections["name"] == "D20").sum()
                    )
                    d40_count = int(
                        (detections["name"] == "D40").sum()
                    )

                    st.metric("D20 Damage", d20_count)
                    st.metric("D40 Damage", d40_count)

                    if len(detections) > 0:

                        st.write("**Detection Details**")
                        st.dataframe(
                            detections[
                                [
                                    "name",
                                    "confidence",
                                    "xmin",
                                    "ymin",
                                    "xmax",
                                    "ymax"
                                ]
                            ],
                            use_container_width=True
                        )

                    else:
                        st.info(
                            "No damage detected above the confidence threshold."
                        )

                    # ---------------- VOICE ASSISTANT ----------------
                    voice_text = (
                        "Road damage detection complete. "
                        f"Total {len(detections)} damaged areas detected. "
                        f"{d20_count} D20 damages and "
                        f"{d40_count} D40 damages detected."
                    )

                    st.subheader("🔊 Voice Assistant")
                    st.caption(
                        "Press the button to hear the detection summary."
                    )

                    components.html(
                        f"""
                        <button onclick="speakResult()"
                        style="
                            padding: 10px 16px;
                            font-size: 16px;
                            cursor: pointer;
                            border-radius: 8px;
                            border: 1px solid #888;
                        ">
                            🔊 Speak Results
                        </button>

                        <script>
                        function speakResult() {{
                            const text = {json.dumps(voice_text)};
                            const synth = window.speechSynthesis;
                            const msg = new SpeechSynthesisUtterance(text);

                            const chooseVoice = () => {{
                                const voices = synth.getVoices();

                                const preferredVoice = voices.find(v =>
                                    /^en/i.test(v.lang) &&
                                    /zira|samantha|karen|susan|aria|jenny|female/i.test(v.name)
                                );

                                const englishVoice = voices.find(v =>
                                    /^en/i.test(v.lang)
                                );

                                const selectedVoice =
                                    preferredVoice || englishVoice;

                                if (selectedVoice) {{
                                    msg.voice = selectedVoice;
                                    msg.lang = selectedVoice.lang;
                                }} else {{
                                    msg.lang = "en-US";
                                }}

                                synth.cancel();
                                synth.speak(msg);
                            }};

                            if (synth.getVoices().length > 0) {{
                                chooseVoice();
                            }} else {{
                                synth.onvoiceschanged = chooseVoice;
                            }}
                        }}
                        </script>
                        """,
                        height=65
                    )

                    # ---------------- LOCATION LINK ----------------
                    if latitude is not None and longitude is not None:
                        st.markdown(
                            f"[📍 View captured location on Google Maps]"
                            f"(https://www.google.com/maps?q={latitude},{longitude})"
                        )

                    # Keep detection results for report download
                    st.session_state["road_detections"] = detections
                    st.session_state["road_image_name"] = getattr(
                        uploaded_file, "name", "camera_photo.jpg"
                    )
                    st.session_state["road_latitude"] = latitude
                    st.session_state["road_longitude"] = longitude
                    st.session_state["road_detection_time"] = (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )

            except Exception as e:
                st.error(f"Detection failed: {e}")


# ---------------- DOWNLOAD DETECTION REPORT ----------------
st.subheader("📋 Detection Report")

if "road_detections" in st.session_state:

    report = st.session_state["road_detections"].copy()

    if len(report) > 0:

        report["date_time"] = st.session_state.get(
            "road_detection_time"
        )
        report["image_name"] = st.session_state.get(
            "road_image_name"
        )
        report["latitude"] = st.session_state.get(
            "road_latitude"
        )
        report["longitude"] = st.session_state.get(
            "road_longitude"
        )

        csv_data = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Detection Report (CSV)",
            data=csv_data,
            file_name="road_damage_report.csv",
            mime="text/csv"
        )

    else:
        st.info("No road damage detected in this image.")
