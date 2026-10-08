"""
OZAIANet - Thermal Image-Based Diabetes Classification
Production-ready Streamlit Web Application with Legal & Research Consent Gate
"""

import os
import time
from pathlib import Path
from PIL import Image
import streamlit as st
import tensorflow as tf

from utils import (
    validate_thermal_foot_image,
    get_sample_images,
    predict_image,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OZAIANet | Thermal Foot Diabetes Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING (SERIOUS RESEARCH / CLINICAL DESIGN)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Typography & Font Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1200px;
    }
    
    /* Primary Action Buttons - Highlighted Clinical Green */
    div[data-testid="stButton"] button[kind="primary"],
    .stButton button[kind="primary"] {
        background-color: #059669 !important;
        background-image: linear-gradient(180deg, #10B981 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #047857 !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.65rem 1.4rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:hover,
    .stButton button[kind="primary"]:hover {
        background-color: #047857 !important;
        background-image: linear-gradient(180deg, #059669 0%, #047857 100%) !important;
        border-color: #065F46 !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:disabled,
    .stButton button[kind="primary"]:disabled {
        background: #334155 !important;
        background-image: none !important;
        color: #64748B !important;
        border: 1px solid #475569 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
        transform: none !important;
    }
    
    /* Secondary Action Buttons */
    div[data-testid="stButton"] button[kind="secondary"],
    .stButton button[kind="secondary"] {
        background: #1E293B !important;
        color: #CBD5E1 !important;
        border: 1px solid #475569 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div[data-testid="stButton"] button[kind="secondary"]:hover,
    .stButton button[kind="secondary"]:hover {
        background: #334155 !important;
        color: #F8FAFC !important;
        border-color: #64748B !important;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, rgba(14, 25, 45, 0.95) 0%, rgba(20, 36, 68, 0.95) 100%);
        border: 1px solid rgba(80, 120, 200, 0.25);
        border-radius: 14px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 500;
        color: #E2E8F0;
        margin-bottom: 0.8rem;
    }
    
    .hero-desc {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.6;
        max-width: 850px;
        margin-bottom: 0;
    }
    
    /* Model Architecture Pill */
    .arch-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(96, 165, 250, 0.12);
        color: #93C5FD;
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 0.3rem 0.85rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 1rem;
    }
    
    /* Result Cards */
    .result-box-dm {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(185, 28, 28, 0.08) 100%);
        border: 1.5px solid rgba(239, 68, 68, 0.5);
        border-radius: 14px;
        padding: 1.8rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .result-box-ctrl {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 1.5px solid rgba(16, 185, 129, 0.5);
        border-radius: 14px;
        padding: 1.8rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .result-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    
    .result-class {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }
    
    .result-prob {
        font-size: 1.35rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Disclaimer Footer */
    .disclaimer-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        margin-top: 3rem;
        font-size: 0.85rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "disclaimer_agreed" not in st.session_state:
    st.session_state.disclaimer_agreed = False
if "disclaimer_declined" not in st.session_state:
    st.session_state.disclaimer_declined = False
if "disclaimer_start_time" not in st.session_state:
    st.session_state.disclaimer_start_time = time.time()

if "active_source" not in st.session_state:
    st.session_state.active_source = "sample"  # 'sample' or 'upload'
if "active_sample_index" not in st.session_state:
    st.session_state.active_sample_index = 1
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# =============================================================================
# MANDATORY LEGAL & RESEARCH DISCLAIMER GATE
# =============================================================================
if not st.session_state.disclaimer_agreed:
    # --- IF USER PREVIOUSLY DECLINED ---
    if st.session_state.disclaimer_declined:
        with st.container(border=True):
            st.error(
                "### Access Denied — Terms Not Accepted\n\n"
                "You have declined the mandatory Research and Legal Disclaimer Agreement.\n\n"
                "Access to the OZAIANet model, inference pipeline, and research interface is strictly prohibited without explicit acceptance of these terms.\n\n"
                "**Please leave this site and close your browser window immediately.**"
            )
        
        col_c1, col_c2, col_c3 = st.columns([1, 1.2, 1])
        with col_c2:
            if st.button("Review Terms and Reconsider", use_container_width=True, type="secondary"):
                st.session_state.disclaimer_declined = False
                st.session_state.disclaimer_start_time = time.time()
                st.rerun()
        st.stop()

    # --- DISCLAIMER REVIEW SCREEN (CLEAN NATIVE STREAMLIT CONTAINERS) ---
    with st.container(border=True):
        st.markdown("## MANDATORY LEGAL & RESEARCH DISCLAIMER")
        st.caption("Please review all technical conditions, scientific limitations, and liability terms carefully before proceeding.")
        st.markdown("---")

        with st.container(height=400):
            st.markdown("#### 1. Academic Research Prototype & Dataset Distribution Gap")
            st.markdown(
                "This application and the underlying deep neural network (**OZAIANet**) are products of academic research "
                "developed and evaluated strictly on clean, curated, and standardized benchmark plantar thermogram datasets. "
                "External, user-uploaded images may vary drastically in sensor calibration, thermal sensitivity, ambient temperature, "
                "emissivity, field-of-view, and subject preparation. **Consequently, predictions on custom-uploaded images may deviate "
                "substantially and can be completely incorrect or misleading.**"
            )

            st.markdown("#### 2. Algorithmic Pre-Inference Mask / Heuristic Safety Layer")
            st.markdown(
                "A computer vision validation mask layer is prepended to the model to intercept non-pertinent inputs and assess "
                "whether an uploaded image conforms to the structural scope of the model (segmented sole silhouette, dark borders, "
                "false-color thermal palette). **This validation layer is a heuristic filter and is NOT infallible.** It may inadvertently "
                "permit out-of-distribution imagery or reject valid scans. Successful validation does NOT guarantee diagnostic relevance."
            )

            st.markdown("#### 3. Strict Non-Medical Diagnosis & Mandatory Physician Consultation")
            st.markdown(
                "**UNDER NO CIRCUMSTANCES DOES THIS APPLICATION PROVIDE A MEDICAL DIAGNOSIS, PROGNOSIS, OR CLINICAL ADVICE.** "
                "The output labels (*DM / CTRL*) and model probabilities represent raw statistical correlations learned by an "
                "experimental algorithm and must NEVER be believed, interpreted, or relied upon as clinical fact. "
                "If you have any medical symptoms, suspected diabetic foot ulcers, neuropathy, or vascular conditions, "
                "**you must immediately consult a qualified, licensed medical physician or podiatric specialist.**"
            )

            st.markdown("#### 4. Incompatibility with Standard Smartphone RGB Images")
            st.markdown(
                "This deep-learning model is specifically designed for radiometric false-color thermograms. "
                "**IT DOES NOT WORK ON STANDARD RGB PHOTOGRAPHS CAPTURED BY SMARTPHONES, CONSUMER CAMERAS, OR WEB-CAMS.** "
                "Traditional optical photographs lack temperature distribution data; submitting smartphone photos will yield invalid "
                "results or trigger validation rejection."
            )

            st.error(
                "**5. COMPLETE LIMITATION OF LIABILITY & HOLD HARMLESS:**\n\n"
                "IN CASE OF ANY ERRORS, INACCURACIES, MISCLASSIFICATIONS, HEALTH-RELATED OUTCOMES, OR DAMAGES OF ANY NATURE "
                "(WHETHER DIRECT, INDIRECT, OR CONSEQUENTIAL) ARISING OUT OF OR IN CONNECTION WITH THE USE OF THIS SOFTWARE, "
                "ITS PREDICTIONS, OR INABILITY TO USE THIS SERVICE, **THE CREATOR, AUTHORS, RESEARCHERS, AND AFFILIATED ENTITIES "
                "SHALL NOT BE HELD RESPONSIBLE OR LIABLE UNDER ANY CIRCUMSTANCES OR LEGAL THEORY.**"
            )

            st.markdown("#### 6. Mandatory Consent or Site Exit")
            st.markdown(
                "Full agreement to each and every term stated above is an absolute mandatory condition of use. "
                "If you do not agree with all of these terms in their entirety, **do not proceed** and exit this application immediately."
            )

    # 5-SECOND READING TIMER ENFORCEMENT
    elapsed_seconds = time.time() - st.session_state.disclaimer_start_time
    remaining_seconds = 5 - int(elapsed_seconds)

    if remaining_seconds > 0:
        timer_placeholder = st.empty()
        for s in range(remaining_seconds, 0, -1):
            timer_placeholder.warning(
                f"[MANDATORY REVIEW PERIOD] Please read the legal terms above carefully. "
                f"Acknowledgement unlocks in {s} second{'s' if s > 1 else ''}..."
            )
            time.sleep(1)
        timer_placeholder.empty()
        st.rerun()

    # AFTER 5 SECONDS: Enable consent controls
    st.info("[REVIEW PERIOD COMPLETED] Please confirm your acknowledgement below to proceed.")

    agreed_check = st.checkbox(
        "I confirm that I have read, understood, and agree to all the terms, limitations, and liabilities listed above. "
        "I acknowledge that OZAIANet is an experimental research demonstration and NOT a medical diagnostic tool.",
        key="disclaimer_checkbox",
    )

    col_btn_agree, col_btn_decline = st.columns([1.2, 1])

    with col_btn_agree:
        if st.button("I Agree and Proceed to OZAIANet", type="primary", disabled=not agreed_check, use_container_width=True):
            st.session_state.disclaimer_agreed = True
            st.rerun()

    with col_btn_decline:
        if st.button("Decline and Exit Application", type="secondary", use_container_width=True):
            st.session_state.disclaimer_declined = True
            st.rerun()

    st.caption("Notice: If you do not agree with these terms, please leave this site immediately by closing your browser tab.")

    # Stop execution completely so NO application layout or model is accessible
    st.stop()


# =============================================================================
# MAIN APPLICATION LAYOUT (ONLY ACCESSIBLE AFTER EXPLICIT CONSENT)
# =============================================================================

# -----------------------------------------------------------------------------
# CACHED MODEL LOADER
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ozaianet_model(model_path: str = "best_model_2.h5"):
    """
    Load the trained OZAIANet model once and cache it in memory.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights file '{model_path}' was not found in the root directory. "
            "Please ensure 'best_model_2.h5' is present."
        )
    model = tf.keras.models.load_model(model_path)
    return model


# Discover samples
sample_list = get_sample_images("samples")

# -----------------------------------------------------------------------------
# SIDEBAR: MODEL & METHODOLOGY INFORMATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("OZAIANet Research")
    st.markdown(
        """
        **OZAIANet** is a dual-backbone convolutional neural network designed 
        for classifying diabetic microvascular alterations from plantar thermograms.
        """
    )
    
    st.markdown("### Model Architecture")
    st.markdown(
        """
        - **Backbone 1**: DenseNet121
        - **Backbone 2**: EfficientNetV2B0
        - **Input Shape**: `224 × 224 × 3` (RGB)
        - **Output**: Sigmoid probability `[0.0, 1.0]`
        - **Threshold**: `0.50`
        """
    )
    
    st.markdown("### Classification Categories")
    st.markdown(
        """
        - **DM** (Diabetic Microvascular Pattern): Probability `> 0.50`
        - **CTRL** (Control / Normal Pattern): Probability `≤ 0.50`
        """
    )

    st.markdown("### Validation Safety Gate")
    st.markdown(
        """
        Input images pass through a heuristic safety check before entering the model:
        1. Valid channels and dimensions
        2. Segmented dark background (>12%)
        3. Dark perimeter margins (>35%)
        4. Dominant foreground sole (>55%)
        5. False-color thermal variance (>22)
        6. Cohesive sole contour (solidity > 0.65)
        """
    )
    
    st.markdown("---")
    st.caption("Status: Terms Acknowledged")
    if st.button("Revisit / Revoke Legal Disclaimer", use_container_width=True, type="secondary"):
        st.session_state.disclaimer_agreed = False
        st.session_state.disclaimer_start_time = time.time()
        st.rerun()

# -----------------------------------------------------------------------------
# HERO HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">OZAIANet</div>
        <div class="hero-subtitle">Thermal Image-Based Diabetes Classification</div>
        <p class="hero-desc">
            Deep-learning evaluation of sole-of-foot plantar thermal scans. 
            Select one of the validated research samples or upload an external thermal scan to 
            evaluate classification probabilities.
        </p>
        <div class="arch-badge">
            Dual-Backbone: DenseNet121 + EfficientNetV2B0 | 224x224 RGB
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# LOAD MODEL (CACHED)
# -----------------------------------------------------------------------------
try:
    with st.spinner("Initializing OZAIANet neural network weights..."):
        model = load_ozaianet_model("best_model_2.h5")
except Exception as e:
    st.error(f"[ERROR] Failed to load model weights: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# IMAGE INPUT SECTION: TABS (SAMPLE SELECTOR vs UPLOAD)
# -----------------------------------------------------------------------------
st.subheader("Select or Upload Input Image")

tab_samples, tab_upload = st.tabs([
    "Benchmark Samples (pic1 - pic6)",
    "Upload Custom Image"
])

current_image = None
current_image_name = None
current_source_type = None

# --- TAB 1: SAMPLES GRID ---
with tab_samples:
    st.markdown("Choose one of the 6 benchmark plantar thermal scans from the research dataset:")
    
    if not sample_list:
        st.warning("No sample images were found in the samples/ directory.")
    else:
        # Render a clean responsive 6-column grid
        cols = st.columns(len(sample_list))
        for idx, (label, fname, fpath) in enumerate(sample_list):
            sample_num = idx + 1
            is_active = (
                st.session_state.active_source == "sample"
                and st.session_state.active_sample_index == sample_num
            )
            
            with cols[idx]:
                try:
                    thumb_img = Image.open(fpath)
                    st.image(thumb_img, use_container_width=True)
                    btn_label = f"[Selected] {fname}" if is_active else f"Select {fname}"
                    btn_type = "primary" if is_active else "secondary"
                    
                    if st.button(btn_label, key=f"btn_sample_{sample_num}", type=btn_type, use_container_width=True):
                        st.session_state.active_source = "sample"
                        st.session_state.active_sample_index = sample_num
                        st.rerun()
                except Exception as err:
                    st.caption(f"Error: {err}")

# --- TAB 2: FILE UPLOAD ---
with tab_upload:
    st.markdown("Upload a thermal sole-of-foot scan (PNG, JPG/JPEG, or WEBP):")
    uploaded_file = st.file_uploader(
        "Upload thermal image file",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="custom_file_uploader",
    )
    
    if uploaded_file is not None:
        if st.session_state.last_uploaded_name != uploaded_file.name:
            st.session_state.active_source = "upload"
            st.session_state.last_uploaded_name = uploaded_file.name

# -----------------------------------------------------------------------------
# RESOLVE ACTIVE IMAGE
# -----------------------------------------------------------------------------
if st.session_state.active_source == "upload" and uploaded_file is not None:
    try:
        current_image = Image.open(uploaded_file)
        current_image_name = uploaded_file.name
        current_source_type = "Uploaded File"
    except Exception as e:
        st.error(f"Could not open uploaded file: {e}")
elif sample_list:
    sample_idx = min(st.session_state.active_sample_index - 1, len(sample_list) - 1)
    label, fname, fpath = sample_list[sample_idx]
    current_image = Image.open(fpath)
    current_image_name = fname
    current_source_type = f"Benchmark Sample ({fname})"
else:
    current_image = None

# -----------------------------------------------------------------------------
# PREVIEW & ANALYSIS WORKFLOW
# -----------------------------------------------------------------------------
st.markdown("---")

if current_image is None:
    st.info("Please select a benchmark sample or upload an image to begin analysis.")
    st.stop()

col_preview, col_analysis = st.columns([1, 1.4], gap="large")

with col_preview:
    st.markdown("### Selected Image")
    st.image(current_image, caption=f"{current_source_type} ({current_image_name})", use_container_width=True)
    
    img_w, img_h = current_image.size
    st.caption(f"**Resolution**: {img_w} × {img_h} px | **Mode**: {current_image.mode}")

with col_analysis:
    st.markdown("### Validation and Model Inference")

    # -------------------------------------------------------------------------
    # SAFETY GATE VALIDATION
    # -------------------------------------------------------------------------
    validation_result = validate_thermal_foot_image(current_image)
    
    if not validation_result.is_valid:
        # VALIDATION FAILED: Block inference
        st.error(
            f"**[VALIDATION REJECTION] Invalid Input Image**\n\n"
            f"Please upload a thermal image of the sole of a foot similar to the benchmark examples provided.\n\n"
            f"**Rejection reason:** {validation_result.reason}"
        )
        
        with st.expander("Validation Diagnostics", expanded=True):
            st.markdown(
                "The safety gate rejected this input because it does not exhibit the physical and "
                "geometric characteristics of a segmented foot-sole thermal scan:"
            )
            for k, v in validation_result.details.items():
                st.write(f"- **{k.replace('_', ' ').capitalize()}**: `{v}`")
                
            st.info(
                "**Required Input Profile:**\n"
                "- Segmented dark background surrounding the foot (>12%)\n"
                "- Dark outer border margins (>35%)\n"
                "- Single dominant foot sole shape (>55% of foreground)\n"
                "- Multi-color false-color thermal gradient (variance >22)"
            )
            
        st.warning("[HALTED] Model execution stopped. OZAIANet does not process unverified inputs.")

    else:
        # VALIDATION PASSED: Run OZAIANet inference
        st.success("[SAFETY GATE PASSED] Input verified as a valid thermal foot-sole scan profile.")
        
        with st.expander("Validation Diagnostics (Passed)", expanded=False):
            st.markdown("The heuristic validation gate verified the following image metrics:")
            cols_d = st.columns(2)
            for i, (k, v) in enumerate(validation_result.details.items()):
                with cols_d[i % 2]:
                    st.metric(label=k.replace('_', ' ').title(), value=str(v))

        # Run inference using single standard pipeline
        with st.spinner("Executing OZAIANet inference..."):
            prob, pred_class = predict_image(current_image, model)

        prob_pct = prob * 100.0

        # Render Result Box
        if pred_class == "DM":
            st.markdown(
                f"""
                <div class="result-box-dm">
                    <div class="result-header" style="color: #F87171;">Model Prediction</div>
                    <div class="result-class" style="color: #EF4444;">DM (Diabetic Pattern)</div>
                    <div class="result-prob" style="color: #FCA5A5;">Probability: {prob_pct:.2f}%</div>
                    <p style="color: #FECACA; font-size: 0.95rem; margin-top: 0.8rem; margin-bottom: 0;">
                        The model identified thermal distribution characteristics consistent 
                        with the diabetic group (classification threshold: <strong>&gt; 0.50</strong>).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-box-ctrl">
                    <div class="result-header" style="color: #34D399;">Model Prediction</div>
                    <div class="result-class" style="color: #10B981;">CTRL (Control Pattern)</div>
                    <div class="result-prob" style="color: #6EE7B7;">Probability: {prob_pct:.2f}%</div>
                    <p style="color: #A7F3D0; font-size: 0.95rem; margin-top: 0.8rem; margin-bottom: 0;">
                        The model identified thermal distribution characteristics consistent 
                        with the non-diabetic control group (classification threshold: <strong>≤ 0.50</strong>).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visual Probability Indicator Bar
        st.markdown("**Probability Spectrum (0.00 – 1.00):**")
        st.progress(float(prob))

        col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
        with col_b1:
            st.caption("0.0% (Strong CTRL)")
        with col_b2:
            st.caption("[ 50.0% Decision Boundary ]")
        with col_b3:
            st.caption(f"100.0% (Strong DM) — **Current: {prob_pct:.2f}%**")

# -----------------------------------------------------------------------------
# RESEARCH DISCLAIMER FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-card">
        <strong>RESEARCH & CLINICAL DISCLAIMER:</strong><br>
        This application demonstrates the <strong>OZAIANet</strong> research model and is not intended 
        for medical diagnosis, clinical screening, or treatment decision-making. Predictions are generated 
        by an experimental deep-learning model evaluated on plantar thermograms. Always consult a certified 
        physician or medical specialist for clinical evaluation and diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)
