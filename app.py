import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Drone Detection System",
    page_icon="🚁",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
try:
    model = YOLO("best.pt")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🚁 Drone Detection")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Upload an image containing Birds or Drones "
    "and click Detect Objects."
)

# -----------------------------
# Main UI
# -----------------------------
st.title("🚁 AI Drone Detection System")
st.write(
    "Detect Birds and Drones using a custom-trained YOLOv8 model."
)

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    if st.button("🔍 Detect Objects"):

        with st.spinner("Detecting..."):

            results = model(
                image,
                conf=confidence_threshold
            )

            detections = results[0].boxes

            annotated_img = results[0].plot()

            # -----------------------------
            # Side by Side Display
            # -----------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(
                    image,
                    use_container_width=True
                )

            with col2:
                st.subheader("Detection Result")
                st.image(
                    annotated_img,
                    use_container_width=True
                )

            # -----------------------------
            # Statistics
            # -----------------------------
            total_objects = len(detections)

            drone_count = 0
            bird_count = 0

            for box in detections:

                cls = int(box.cls[0])

                if model.names[cls] == "Drone":
                    drone_count += 1

                elif model.names[cls] == "Bird":
                    bird_count += 1

            st.subheader("📊 Detection Statistics")

            stat1, stat2, stat3 = st.columns(3)

            stat1.metric(
                "Total Objects",
                total_objects
            )

            stat2.metric(
                "Drones",
                drone_count
            )

            stat3.metric(
                "Birds",
                bird_count
            )

            # -----------------------------
            # Detection Details Table
            # -----------------------------
            if total_objects > 0:

                data = []

                for box in detections:

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    data.append({
                        "Object": model.names[cls],
                        "Confidence (%)": round(conf * 100, 2)
                    })

                st.subheader("📋 Detection Details")

                df = pd.DataFrame(data)

                st.dataframe(
                    df,
                    use_container_width=True
                )

                # -----------------------------
                # Detection Summary
                # -----------------------------
                st.subheader("✅ Detection Summary")

                for box in detections:

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    st.success(
                        f"{model.names[cls]} detected "
                        f"({conf:.2%} confidence)"
                    )

            else:
                st.warning(
                    "No Drone or Bird detected."
                )

            # -----------------------------
            # Download Result
            # -----------------------------
            output_path = "detected.jpg"

            cv2.imwrite(
                output_path,
                cv2.cvtColor(
                    annotated_img,
                    cv2.COLOR_RGB2BGR
                )
            )

            with open(output_path, "rb") as file:

                st.download_button(
                    label="📥 Download Result",
                    data=file,
                    file_name="detected.jpg",
                    mime="image/jpeg"
                )