"""Streamlit Web Application for Bird Species Classification.

Interactive visual interface for testing the custom PyTorch CNN model.
Runs in-memory predictions without writing shared temporary files to disk.
"""

import os
import sys
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import BirdPredictor

st.set_page_config(
    page_title="Bird Species Classifier (PyTorch CNN)",
    page_icon="🦜",
    layout="wide"
)

# Custom CSS for polished academic demonstration
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.2rem;
    }
    .pred-species {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .pred-conf {
        font-size: 1.15rem;
        opacity: 0.95;
    }
    .disclaimer-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #374151;
        border-radius: 4px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(checkpoint_path: str):
    """Load and cache the BirdPredictor instance."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        return BirdPredictor(checkpoint_path=checkpoint_path, device_setting="auto")
    except Exception as e:
        st.error(f"Failed to load model checkpoint: {e}")
        return None


def main():
    st.markdown('<div class="main-header">🦅 Bird Species Classification Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Academic Machine Learning Project: Custom 4-Stage CNN trained on '
        '10 species from the Hugging Face Bird Species Dataset.</div>',
        unsafe_allow_html=True
    )

    checkpoint_rel = "models/best_model.pth"
    checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_rel)

    # Sidebar specifications
    st.sidebar.title("Model & System Info")
    st.sidebar.markdown("**Architecture**: Custom 4-Stage CNN")
    st.sidebar.markdown("**Input Resolution**: 224 x 224 RGB")
    st.sidebar.markdown("**Optimization**: AdamW + ReduceLROnPlateau")
    st.sidebar.markdown("**Regularization**: Dropout (p=0.4), Weight Decay")

    if os.path.exists(checkpoint_path):
        st.sidebar.success("✓ Trained Checkpoint Loaded (Epoch 11)")
    else:
        st.sidebar.error("✗ Checkpoint Missing")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Selected 10 Species:")
    species_list = [
        "ABBOTTS BABBLER",
        "ABBOTTS BOOBY",
        "ABYSSINIAN GROUND HORNBILL",
        "AFRICAN CROWNED CRANE",
        "AFRICAN EMERALD CUCKOO",
        "AFRICAN FIREFINCH",
        "AFRICAN OYSTER CATCHER",
        "AFRICAN PIED HORNBILL",
        "AFRICAN PYGMY GOOSE",
        "ALBATROSS"
    ]
    for sp in species_list:
        st.sidebar.markdown(f"- `{sp}`")

    # Guard: Missing checkpoint
    if not os.path.exists(checkpoint_path):
        st.warning(
            "⚠️ **Model Checkpoint Not Found!**\n\n"
            f"Expected checkpoint file at `{checkpoint_path}`.\n\n"
            "To train the model and generate the checkpoint, run:\n"
            "```bash\n"
            "python scripts/prepare_data.py\n"
            "python src/train.py configs/config.yaml\n"
            "```"
        )
        return

    predictor = load_predictor(checkpoint_path)
    if predictor is None:
        st.error("Model predictor could not be initialized.")
        return

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Input Image Selection")
        uploaded_file = st.file_uploader(
            "Upload a bird photograph (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"]
        )

        sample_btn = st.button("Load Test Set Sample")
        image_to_process = None
        source_description = ""

        if uploaded_file is not None:
            try:
                image_to_process = Image.open(uploaded_file).convert("RGB")
                source_description = f"Uploaded File: {uploaded_file.name}"
                st.image(image_to_process, caption=source_description, use_container_width=True)
            except Exception as e:
                st.error(f"Error reading uploaded image: {e}")
        elif sample_btn:
            test_dir = os.path.join(PROJECT_ROOT, "data", "processed", "test")
            sample_path = None
            if os.path.exists(test_dir):
                for sp in species_list:
                    sp_dir = os.path.join(test_dir, sp)
                    if os.path.exists(sp_dir) and os.listdir(sp_dir):
                        sample_path = os.path.join(sp_dir, os.listdir(sp_dir)[0])
                        break
            if sample_path and os.path.exists(sample_path):
                try:
                    image_to_process = Image.open(sample_path).convert("RGB")
                    source_description = f"Test Sample: {os.path.basename(sample_path)} ({os.path.basename(os.path.dirname(sample_path))})"
                    st.image(image_to_process, caption=source_description, use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading sample image: {e}")
            else:
                st.info("Test set samples not found in data/processed/test. Please upload an image.")

    with col2:
        st.subheader("2. Model Classification")
        if image_to_process is not None:
            with st.spinner("Executing Convolutional feature extraction and Softmax classification..."):
                try:
                    # In-memory prediction without writing temporary files to disk
                    preds = predictor.predict(image_to_process, top_k=3)
                    top1 = preds[0]

                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; opacity:0.9;">Top-1 Predicted Species</div>
                            <div class="pred-species">{top1['species']}</div>
                            <div class="pred-conf">Model Probability: <strong>{top1['percentage']}</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("#### Top-3 Predicted Class Probabilities:")
                    for item in preds:
                        col_label, col_val = st.columns([3, 1])
                        with col_label:
                            st.write(f"**#{item['rank']} {item['species']}**")
                        with col_val:
                            st.write(f"{item['percentage']}")
                        st.progress(min(max(item["probability"], 0.0), 1.0))

                    st.markdown(
                        """
                        <div class="disclaimer-box">
                            <strong>Academic Context Note:</strong> This model was trained exclusively on the 10 selected species
                            from the Hugging Face dataset. Displayed probabilities represent normalized Softmax scores across
                            these 10 classes and should not be interpreted as open-domain identification or calibrated Bayesian confidence.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Inference error: {e}")
        else:
            st.info("Upload an image or click 'Load Test Set Sample' on the left to obtain predictions.")


if __name__ == "__main__":
    main()
