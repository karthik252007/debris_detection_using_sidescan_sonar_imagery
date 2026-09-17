import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Debris Detection in Sonar Images",
    page_icon="🌊",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌊 Debris Detection in Sonar Images")
st.write("YOLO-based detection of objects in sonar images.")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("debris.pt")


model = load_model()

# --------------------------------------------------
# Confidence slider
# --------------------------------------------------

confidence = st.slider(
    "Detection Confidence",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a sonar image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Detection
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    # Two columns
    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # Original image
    # --------------------------------------------------

    with col1:

        st.subheader("Original Sonar Image")

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------

    results = model.predict(
        source=np.array(image),
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # YOLO annotated image
    annotated_image = result.plot()

    # --------------------------------------------------
    # Detection image
    # --------------------------------------------------

    with col2:

        st.subheader("Detection Result")

        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )

    # --------------------------------------------------
    # Detection information
    # --------------------------------------------------

    st.divider()

    st.subheader("Detection Information")

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:

        st.success(
            f"{len(boxes)} object(s) detected"
        )

        for i, box in enumerate(boxes):

            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            class_name = model.names[class_id]

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1

            area = width * height

            st.markdown(
                f"""
                ### Object {i + 1}

                **Class:** {class_name}

                **Confidence:** {confidence_score:.2f}

                **Bounding Box:**
                - X1: {x1:.1f}
                - Y1: {y1:.1f}
                - X2: {x2:.1f}
                - Y2: {y2:.1f}

                **Width:** {width:.1f} px

                **Height:** {height:.1f} px

                **Bounding Box Area:** {area:.1f} px²
                """
            )

    else:

        st.warning(
            "No objects detected. Try lowering the confidence threshold."
        )