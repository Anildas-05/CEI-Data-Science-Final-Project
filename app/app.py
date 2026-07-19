# ==========================================================
# SATELLITE LAND-USE CLASSIFICATION &
# TEMPORAL CHANGE DETECTION SYSTEM
# ==========================================================

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import json
import torch
import streamlit as st
from PIL import Image

from models import load_model, get_feature_extractor

from utils import (
    predict_image,
    extract_embedding,
    calculate_similarity,
    transform
)

from heatmap import generate_heatmap


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Satellite Land-Use Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* Hide Streamlit Default Items */





/* Main Page */

.main{
background:#F4F7FB;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
padding-left:2rem;
padding-right:2rem;
}

/* Header */

.header{

background:linear-gradient(90deg,#0F172A,#1D4ED8);

padding:28px;

border-radius:15px;

text-align:center;

color:white;

margin-bottom:25px;

box-shadow:0px 4px 15px rgba(0,0,0,0.25);

}

/* Cards */

.metric-card{

background:white;

padding:20px;

border-radius:15px;

box-shadow:0px 2px 12px rgba(0,0,0,0.08);

margin-bottom:15px;

}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#black;

}

/* Upload Box */

[data-testid="stFileUploader"]{

border:2px dashed #2563EB;

border-radius:15px;

padding:15px;

background:#F8FAFC;

}

/* Footer */

.footer{

text-align:center;

color:gray;

padding:20px;

font-size:16px;

}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="header">

<h1>🛰️ Satellite Land-Use Classification &
Temporal Change Detection</h1>

<h4>
Deep Learning | ResNet18 | Transfer Learning |
Cosine Similarity | Streamlit Dashboard
</h4>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Settings")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    [
        "EuroSAT",
        "UC Merced"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("🧠 Development")

st.sidebar.markdown("""

- ResNet18 CNN

- Transfer Learning

- Deep Feature Embeddings

- Cosine Similarity

- PyTorch

- Streamlit

""")

st.sidebar.markdown("---")

st.sidebar.subheader("✨ Key Features")

st.sidebar.markdown("""

✅ Land-Use Classification

✅ Temporal Change Detection

✅ Feature Embedding Extraction

✅ Confidence Score

✅ Heatmap Visualization

""")

st.sidebar.markdown("---")


# ==========================================================
# LOAD THRESHOLDS
# ==========================================================

threshold_path = os.path.join(
    "models",
    "threshold.json"
)

with open(threshold_path, "r") as f:

    threshold = json.load(f)

CHANGE_THRESHOLD = threshold["change_threshold"]

SIMILARITY_THRESHOLD = threshold["similarity_threshold"]


# ==========================================================
# LOAD MODEL
# ==========================================================

model, classes = load_model(dataset_name)

feature_extractor = get_feature_extractor(model)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.sidebar.subheader("ℹ️ Model Information")

st.sidebar.write(
    f"**Device :** {device}"
)

st.sidebar.write(
    f"**Similarity Threshold :** {SIMILARITY_THRESHOLD:.3f}"
)

st.sidebar.write(
    f"**Change Threshold :** {CHANGE_THRESHOLD:.3f}"
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
This dashboard classifies satellite images into land-use
categories using a ResNet18 model trained with Transfer
Learning and detects temporal changes using Cosine
Similarity between deep feature embeddings.
"""
)
# ==========================================================
# IMAGE UPLOAD SECTION
# ==========================================================

st.markdown("## 📂 Upload Satellite Images")

st.markdown(
"""
Upload **Before** and **After** satellite images captured at
different time periods. The system will classify each image,
extract deep feature embeddings, calculate cosine similarity,
and detect whether a significant land-use change has occurred.
"""
)

st.markdown("---")


# ==========================================================
# BEFORE & AFTER UPLOAD
# ==========================================================

upload_col1, upload_col2 = st.columns(2)

# ---------------- BEFORE IMAGE ----------------

with upload_col1:

    st.markdown("### 🟢 Before Image")

    before_file = st.file_uploader(
        "Choose BEFORE Image",
        type=["jpg", "jpeg", "png"],
        key="before_image"
    )

# ---------------- AFTER IMAGE ----------------

with upload_col2:

    st.markdown("### 🔵 After Image")

    after_file = st.file_uploader(
        "Choose AFTER Image",
        type=["jpg", "jpeg", "png"],
        key="after_image"
    )


# ==========================================================
# IMAGE PREVIEW
# ==========================================================

before_image = None
after_image = None

if before_file is not None:
    before_image = Image.open(before_file).convert("RGB")

if after_file is not None:
    after_image = Image.open(after_file).convert("RGB")


# ==========================================================
# DISPLAY IMAGE PREVIEW
# ==========================================================

if before_image is not None and after_image is not None:

    st.markdown("---")
    st.subheader("🖼 Uploaded Images")

    preview_col1, preview_col2 = st.columns(2)

    with preview_col1:

        st.image(
            before_image,
            caption="Before Satellite Image",
            use_container_width=True
        )

    with preview_col2:

        st.image(
            after_image,
            caption="After Satellite Image",
            use_container_width=True
        )

else:

    st.info("📌 Please upload both satellite images to continue.")


# ==========================================================
# IMAGE INFORMATION
# ==========================================================

if before_image is not None and after_image is not None:

    st.markdown("---")
    st.subheader("📋 Image Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write("### Before Image")

        st.write(f"**Size:** {before_image.size[0]} × {before_image.size[1]}")

        st.write(f"**Mode:** {before_image.mode}")

    with info_col2:

        st.write("### After Image")

        st.write(f"**Size:** {after_image.size[0]} × {after_image.size[1]}")

        st.write(f"**Mode:** {after_image.mode}")


# ==========================================================
# ANALYZE BUTTON
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

left_space, center_button, right_space = st.columns([2,3,2])

with center_button:

    analyze = st.button(
        "🔍 Analyze Images",
        use_container_width=True,
        type="primary",
        disabled=not (
            before_image is not None and
            after_image is not None
        )
    )


# ==========================================================
# LOADING ANIMATION
# ==========================================================

if analyze:

    progress_bar = st.progress(0)

    status = st.empty()

    status.text("Loading AI model...")
    progress_bar.progress(20)

    status.text("Preprocessing images...")
    progress_bar.progress(40)

    status.text("Extracting deep features...")
    progress_bar.progress(65)

    status.text("Computing similarity...")
    progress_bar.progress(85)

    status.text("Generating results...")
    progress_bar.progress(100)

    status.success("✅ Analysis Completed Successfully!")

    st.markdown("---")
# ==========================================================
# IMAGE ANALYSIS
# ==========================================================

if analyze:

    # ------------------------------------------------------
    # PREPROCESS IMAGES
    # ------------------------------------------------------

    before_tensor = transform(before_image).unsqueeze(0)

    after_tensor = transform(after_image).unsqueeze(0)


    # ------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------

    before_prediction, before_confidence = predict_image(
        model=model,
        image_tensor=before_tensor,
        classes=classes,
        device=device
    )

    after_prediction, after_confidence = predict_image(
        model=model,
        image_tensor=after_tensor,
        classes=classes,
        device=device
    )


    # ------------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------------

    before_embedding = extract_embedding(
        feature_extractor=feature_extractor,
        image_tensor=before_tensor,
        device=device
    )

    after_embedding = extract_embedding(
        feature_extractor=feature_extractor,
        image_tensor=after_tensor,
        device=device
    )


    # ------------------------------------------------------
    # COSINE SIMILARITY
    # ------------------------------------------------------

    similarity = calculate_similarity(
        before_embedding,
        after_embedding
    )


    # ------------------------------------------------------
    # CHANGE DETECTION
    # ------------------------------------------------------

    change_detected = similarity < SIMILARITY_THRESHOLD

    if change_detected:

        status = "🚨 Significant Land-Use Change Detected"

        status_color = "red"

    else:

        status = "✅ No Significant Change Detected"

        status_color = "green"


    # ------------------------------------------------------
    # HEATMAP
    # ------------------------------------------------------

    heatmap = generate_heatmap(
        before_image,
        after_image
    )


    # ------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------

    analysis_result = {

        "before_prediction": before_prediction,

        "before_confidence": before_confidence,

        "after_prediction": after_prediction,

        "after_confidence": after_confidence,

        "similarity": similarity,

         "similarity_threshold": SIMILARITY_THRESHOLD,

         "change_threshold": CHANGE_THRESHOLD,

        "change_detected": change_detected,

        "status": status,

        "heatmap": heatmap

    }


    # ------------------------------------------------------
    # SUCCESS MESSAGE
    # ------------------------------------------------------

    st.success("🎉 Analysis Completed Successfully!")
# ==========================================================
# DISPLAY RESULTS
# ==========================================================

if analyze:

    st.markdown("---")
    st.header("📊 Analysis Results")

    # ======================================================
    # PREDICTION RESULTS
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🟢 Before Image")

        st.metric(
            "Predicted Class",
            analysis_result["before_prediction"]
        )

        st.metric(
            "Confidence",
            f"{analysis_result['before_confidence']*100:.2f}%"
        )

        st.progress(
            analysis_result["before_confidence"]
        )


    with col2:

        st.markdown("### 🔵 After Image")

        st.metric(
            "Predicted Class",
            analysis_result["after_prediction"]
        )

        st.metric(
            "Confidence",
            f"{analysis_result['after_confidence']*100:.2f}%"
        )

        st.progress(
            analysis_result["after_confidence"]
        )


    # ======================================================
    # SIMILARITY ANALYSIS
    # ======================================================

    st.markdown("---")
    st.header("📈 Similarity Analysis")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Similarity Score",
            f"{analysis_result['similarity']:.3f}"
        )

    

    # ======================================================
    # CHANGE DETECTION STATUS
    # ======================================================

    st.markdown("---")
    st.header("🚦 Change Detection Status")

    if analysis_result["change_detected"]:

        st.error(
            analysis_result["status"]
        )

    else:

        st.success(
            analysis_result["status"]
        )


    # ======================================================
    # HEATMAP
    # ======================================================

    st.markdown("---")
    st.header("🔥 Land-Use Change Heatmap")

    st.pyplot(
        analysis_result["heatmap"]
    )


    # ======================================================
    # ANALYSIS SUMMARY
    # ======================================================

    st.markdown("---")
    st.header("📋 Analysis Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write("### Classification")

        st.write(
            f"**Before Image:** {analysis_result['before_prediction']}"
        )

        st.write(
            f"**After Image:** {analysis_result['after_prediction']}"
        )

        st.write(
            f"**Before Confidence:** {analysis_result['before_confidence']*100:.2f}%"
        )

        st.write(
            f"**After Confidence:** {analysis_result['after_confidence']*100:.2f}%"
        )


    with summary_col2:

        st.write("### Similarity")

        st.write(
            f"**Similarity Score:** {analysis_result['similarity']:.3f}"
        )

        st.write(
            f"**Threshold:** {analysis_result['similarity_threshold']:.3f}"
        )

        st.write(
            f"**Status:** {analysis_result['status']}"
        )


    # ======================================================
    # DOWNLOAD REPORT
    # ======================================================

    report = f"""
SATELLITE LAND-USE CLASSIFICATION REPORT

========================================

Dataset:
{dataset_name}

----------------------------------------

Before Image

Prediction:
{analysis_result['before_prediction']}

Confidence:
{analysis_result['before_confidence']*100:.2f}%

----------------------------------------

After Image

Prediction:
{analysis_result['after_prediction']}

Confidence:
{analysis_result['after_confidence']*100:.2f}%

----------------------------------------

Similarity Score:
{analysis_result['similarity']:.3f}

Similarity Threshold:
{analysis_result['similarity_threshold']:.3f}

Change Threshold:
{analysis_result['change_threshold']:.3f}

----------------------------------------

Final Status

{analysis_result['status']}

========================================
"""

    st.download_button(
        "📄 Download Analysis Report",
        report,
        file_name="Satellite_Analysis_Report.txt",
        mime="text/plain"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style="
text-align:center;
padding:20px;
color:gray;
">

<h3>🛰️ Satellite Land-Use Classification & Temporal Change Detection</h3>

<p>
Developed using
<b>PyTorch</b> •
<b>ResNet18</b> •
<b>Transfer Learning</b> •
<b>Cosine Similarity</b> •
<b>Streamlit</b>
</p>

<p>
Final Year B.Tech Project
</p>

</div>
""",
unsafe_allow_html=True
)