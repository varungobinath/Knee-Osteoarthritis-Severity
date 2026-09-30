import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KneeVision | Osteoarthritis Analyzer",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark card style */
    .card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        border: 1px solid #3a3a55;
    }

    /* Grade badge */
    .grade-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6C63FF, #48cae4);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 10px 28px;
        border-radius: 50px;
        margin: 8px 0;
    }

    /* Score row */
    .score-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #333355;
        font-size: 0.95rem;
    }

    .score-label { color: #aab4cc; font-weight: 500; }
    .score-value { color: #e0e6f8; font-family: monospace; font-size: 0.9rem; }
    .score-top   { color: #6C63FF; font-weight: 700; }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121e, #1a1a2e);
        border-right: 1px solid #2a2a40;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 10px 0 4px;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6C63FF, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .main-header p {
        color: #8892a4;
        font-size: 1rem;
        margin-top: 4px;
    }

    /* Hide default streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    from PIL import ImageDraw

    img_h, img_w = img.shape[0], img.shape[1]

    # Resize heatmap to image size to find exact peak pixel location
    heatmap_resized = np.array(
        tf.keras.preprocessing.image.array_to_img(
            np.expand_dims(heatmap, axis=-1)
        ).resize((img_w, img_h))
    ).squeeze()

    # Find top N peak activation points (spread apart, not clustered)
    num_peaks = 3
    min_dist = 30  # minimum pixels apart between peaks
    flat_sorted = np.argsort(heatmap_resized.ravel())[::-1]
    peaks = []
    for idx in flat_sorted:
        py, px = np.unravel_index(idx, heatmap_resized.shape)
        # skip if too close to an already selected peak
        if all(abs(py - y) > min_dist or abs(px - x) > min_dist for y, x in peaks):
            peaks.append((py, px))
        if len(peaks) == num_peaks:
            break

    # Build Grad-CAM overlay
    heatmap_u8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_u8]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_w, img_h))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_pil = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Draw rectangles: RED for top peak, YELLOW for the rest
    box_size = 10  # half-size of the rectangle (20x20 total)
    draw = ImageDraw.Draw(superimposed_pil)
    for i, (peak_y, peak_x) in enumerate(peaks):
        color = (255, 230, 0)  # all yellow
        draw.rectangle(
            [peak_x - box_size, peak_y - box_size, peak_x + box_size, peak_y + box_size],
            outline=color,
            width=2,
        )

    return superimposed_pil


GRADE_INFO = {
    "Healthy":  ("Grade 0", "#2ecc71", "No signs of osteoarthritis detected."),
    "Doubtful": ("Grade 1", "#f1c40f", "Possible minor bone spur growth."),
    "Minimal":  ("Grade 2", "#e67e22", "Definite bone spurs; possible narrowing."),
    "Moderate": ("Grade 3", "#e74c3c", "Moderate joint space narrowing observed."),
    "Severe":   ("Grade 4", "#8e44ad", "Large bone spurs and severe joint narrowing."),
}

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = tf.keras.models.load_model(
        "../src/models/model_Xception_ft.hdf5"
    )

    gm = tf.keras.models.clone_model(m)
    gm.set_weights(m.get_weights())
    gm.layers[-1].activation = None

    gm = tf.keras.models.Model(
        inputs=[gm.inputs],
        outputs=[
            gm.get_layer("global_average_pooling2d_1").input,
            gm.output
        ]
    )

    return m, gm

model, grad_model = load_model()
target_size = (224, 224)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦴 KneeVision")
    st.markdown("**Automated Knee Osteoarthritis Grading Using Deep Convolutional Neural Networks and Interpretability Tools.**")
    st.markdown("---")
    st.markdown("### 📤 Upload X-Ray")
    uploaded_file = st.file_uploader(
        "Choose a knee X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG and PNG formats",
    )
    st.markdown("---")
    st.markdown("### ℹ️ KL Grading Scale")
    for name, (grade, color, desc) in GRADE_INFO.items():
        st.markdown(f"<span style='color:{color}'>●</span> **{grade} – {name}**: {desc}", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model: Xception (fine-tuned) · Balanced Accuracy: 67%")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🦴 KneeVision</h1>
    <p>Knee Osteoarthritis Severity Analyzer using Deep Learning</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Main ──────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload a knee X-ray image from the sidebar to begin analysis.")
    st.markdown("""
    #### How it works
    1. Upload a knee X-ray (JPG / PNG) from the sidebar
    2. Click **Analyze Image**
    3. View the predicted KL severity grade, confidence scores, and Grad-CAM heatmap
    """)
else:
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=target_size)
    img_array_raw = tf.keras.preprocessing.image.img_to_array(img)

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 📷 Uploaded X-Ray")
        st.image(uploaded_file, use_column_width=True)
        analyze = st.button("🔍 Analyze Image")

    if analyze:
        img_input = np.expand_dims(img_array_raw.copy(), axis=0).astype(np.float32)
        img_input = tf.keras.applications.xception.preprocess_input(img_input)

        with st.spinner("Analyzing image with deep learning model..."):
            raw_output = model.predict(img_input)

        raw_scores = raw_output[0]
        predicted_idx = int(np.argmax(raw_scores))
        predicted_class = class_names[predicted_idx]
        grade_label, grade_color, grade_desc = GRADE_INFO[predicted_class]

        with col_result:
            st.markdown("#### ✅ Diagnosis Result")
            st.markdown(f"""
            <div class="card">
                <div style="color:#aab4cc; font-size:0.85rem;">Predicted Severity</div>
                <div class="grade-badge">{predicted_class} ({grade_label})</div>
                <div style="color:{grade_color}; margin-top:8px; font-size:0.95rem;">{grade_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Model Raw Confidence Scores")
            scores_html = ""
            for i, (cls, score) in enumerate(zip(class_names, raw_scores)):
                top = i == predicted_idx
                label_cls = "score-top" if top else "score-label"
                val_cls = "score-top" if top else "score-value"
                star = " ⭐" if top else ""
                scores_html += f"""
                <div class="score-row">
                    <span class="{label_cls}">{cls}{star}</span>
                    <span class="{val_cls}">{float(score)}</span>
                </div>"""
            st.markdown(f'<div class="card">{scores_html}</div>', unsafe_allow_html=True)

        # Grad-CAM
        st.markdown("---")
        st.markdown("#### 🔬 Grad-CAM Heatmap — Region of Interest")
        gcam_col, bar_col = st.columns(2, gap="large")

        with gcam_col:
            heatmap = make_gradcam_heatmap(grad_model, img_input)
            cam_image = save_and_display_gradcam(img_array_raw, heatmap)
            st.image(cam_image, use_column_width=True, caption="Areas influencing the prediction")

        with bar_col:
            pct_scores = raw_scores * 100
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor("#1e1e2f")
            ax.set_facecolor("#1e1e2f")
            colors = ["#6C63FF" if i == predicted_idx else "#3a3a55" for i in range(len(class_names))]
            bars = ax.barh(class_names, pct_scores, height=0.5, color=colors)
            for bar, pct in zip(bars, pct_scores):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.2f}%", va="center", color="#e0e6f8", fontsize=9)
            ax.set_xlabel("Confidence (%)", color="#aab4cc")
            ax.tick_params(colors="#aab4cc")
            ax.spines[:].set_color("#3a3a55")
            ax.set_xlim([0, 115])
            fig.tight_layout()
            st.pyplot(fig)
