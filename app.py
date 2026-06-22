import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_helper import (
    DAMAGE_CONFIDENCE_THRESHOLD,
    VEHICLE_CONFIDENCE_THRESHOLD,
    predict,
    validate_vehicle_image,
)


ALLOWED_FILE_TYPES = ["jpg", "jpeg", "png", "webp"]


st.set_page_config(
    page_title="Vehicle Damage Detection",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 950px;
    }
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 10px;
        border: 1px solid #fecaca;
        border-left: 6px solid #b91c1c;
        background: #fff7f7;
        color: #1f2933;
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        letter-spacing: 0;
    }
    .hero p {
        margin: 0.65rem 0 0;
        color: #5f6b7a;
        font-size: 1rem;
    }
    .result-card {
        border: 1px solid #dbeafe;
        border-radius: 10px;
        padding: 1.25rem;
        background: #f8fbff;
        margin-top: 1rem;
    }
    .result-label {
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .result-class {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .small-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Vehicle Damage Detection</h1>
        <p>
            Upload a clear vehicle image to classify visible front or rear damage.
            The app rejects unrelated images and avoids forced predictions when confidence is low.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How to use")
    st.write("1. Upload a clear vehicle image.")
    st.write("2. Make sure the damaged area is visible.")
    st.write("3. Review the prediction and confidence score.")
    st.divider()
    st.caption(f"Vehicle image threshold: {VEHICLE_CONFIDENCE_THRESHOLD:.0%}")
    st.caption(f"Damage confidence threshold: {DAMAGE_CONFIDENCE_THRESHOLD:.0%}")

st.subheader("Upload Image")
uploaded_file = st.file_uploader(
    "Choose a vehicle damage image",
    type=ALLOWED_FILE_TYPES,
    help="Supported formats: JPG, JPEG, PNG, and WEBP.",
)

if uploaded_file is None:
    st.info("Upload a vehicle image to start the damage check.")
else:
    file_extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if file_extension not in ALLOWED_FILE_TYPES:
        st.error("Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be opened as an image. Please upload a valid image file.")
        st.stop()

    st.image(image, caption="Uploaded image preview", use_container_width=True)

    with st.spinner("Checking whether this looks like a vehicle image..."):
        try:
            vehicle_result = validate_vehicle_image(image)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        except Exception:
            st.error("The app could not validate this image. Please try another image.")
            st.stop()

    if not vehicle_result["is_vehicle"]:
        st.warning(
            "This image does not look like a vehicle damage image. "
            "Please upload a valid vehicle image."
        )
        st.caption(
            f"Vehicle confidence: {vehicle_result['vehicle_confidence']:.1%} "
            f"(required: {vehicle_result['threshold']:.0%})"
        )
        st.stop()

    with st.spinner("Running vehicle damage prediction..."):
        try:
            result = predict(image)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        except Exception:
            st.error("Prediction failed. Please try again with a clearer vehicle image.")
            st.stop()

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-label">Prediction confidence</div>', unsafe_allow_html=True)
    st.progress(result["confidence"])
    st.write(f"{result['confidence']:.1%}")

    if not result["is_confident"]:
        st.warning(
            "The model is not confident about this prediction. "
            "Please upload a clearer vehicle damage image."
        )
        st.caption(f"Required confidence: {result['threshold']:.0%}")
    else:
        st.markdown('<div class="result-label">Predicted damage class</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-class">{result["class_name"]}</div>',
            unsafe_allow_html=True,
        )
        st.success("Prediction completed successfully.")

    with st.expander("Class confidence details"):
        for class_name, confidence in sorted(
            result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            st.write(f"{class_name}: {confidence:.1%}")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <p class="small-note">
    Tip: best results come from clear front or rear vehicle photos where the damaged area is visible.
    This tool is a portfolio demo and should not replace professional inspection.
    </p>
    """,
    unsafe_allow_html=True,
)
