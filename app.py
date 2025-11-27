# app.py — Improved Grad-CAM (robust, smoother, recursive layer lookup)
import io
import traceback
from typing import Optional, Tuple, List

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm

# -------------------------
# Page config + styling (kept from your original)
# -------------------------
st.set_page_config(page_title="Malaria RBC Classifier", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    h1 { color: #ffffff; font-weight: 300; font-size: 2.2rem; text-align: center; margin-bottom: 1rem; }
    .result-card { background: linear-gradient(135deg,#1a1a1a,#0f0f0f); border-radius:12px; padding:1rem; margin:1rem 0; }
    p, span, label { color: #cccccc; }
    img { border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
</style>
""", unsafe_allow_html=True)
st.title("🩸 Malaria RBC Classifier — Grad-CAM (improved)")

# -------------------------
# Config
# -------------------------
MODEL_PATHS = [
    "outputs/model_finetuned.h5",
    "model_finetuned.h5",
    "outputs/best_model.h5",
    "best_model.h5",
]
IMG_SIZE = (224, 224)  # model input expected size (override if your model uses different)
SMOOTH_SAMPLES_DEFAULT = 8
SMOOTH_SIGMA = 0.08  # fraction of image size for jitter

# -------------------------
# Utilities
# -------------------------
@st.cache_resource
def load_model_cached():
    last_err = None
    for p in MODEL_PATHS:
        try:
            if tf.io.gfile.exists(p):
                model = keras.models.load_model(p, compile=False)
                return model, p
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(f"Model not found in any of: {MODEL_PATHS}\nLast error: {last_err}")

def find_layer_recursive(model: keras.Model, layer_name: str) -> Optional[keras.layers.Layer]:
    """
    Find a layer by name across the model including nested models.
    """
    # direct attempt
    try:
        return model.get_layer(layer_name)
    except Exception:
        pass
    # recursive search
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        if isinstance(layer, keras.Model):
            found = find_layer_recursive(layer, layer_name)
            if found is not None:
                return found
    return None

def list_conv_like_layers_recursive(model: keras.Model) -> List[str]:
    """
    Return list of candidate layer names that have 4D outputs (conv-like).
    """
    names = []
    def _recurse(M: keras.Model):
        for L in M.layers:
            try:
                out_shape = getattr(L, "output_shape", None)
                if out_shape is not None and len(out_shape) == 4:
                    names.append(L.name)
            except Exception:
                pass
            if isinstance(L, keras.Model):
                _recurse(L)
    _recurse(model)
    # unique preserving order
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return uniq

def preprocess_for_model(img_pil: Image.Image, model: keras.Model) -> np.ndarray:
    """
    Try to use model-specific preprocessing (if built on known apps), else fallback.
    Returns batch array shape (1,H,W,3)
    """
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    # attempt to find a preprocessing function used by the model
    # common case: EfficientNet / ResNet etc.
    # fall back to EfficientNet preprocess_input as you had
    try:
        # check model._is_compiled_with or architecture hint: look for 'efficientnet' in layer names
        if any("efficientnet" in L.name.lower() for L in model.layers):
            arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        elif any("resnet" in L.name.lower() for L in model.layers):
            arr = tf.keras.applications.resnet.preprocess_input(arr)
        else:
            # last resort: scale to [-1,1]
            arr = (arr / 127.5) - 1.0
    except Exception:
        arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, 0).astype(np.float32)

def overlay_heatmap_pil(original_img: Image.Image, heatmap: np.ndarray, alpha: float=0.45, cmap_name: str="jet") -> Image.Image:
    """
    heatmap: 2D array normalized [0,1]
    """
    if heatmap is None:
        return original_img
    if heatmap.ndim != 2:
        heatmap = np.mean(heatmap, axis=-1)
    hmin, hmax = float(np.nanmin(heatmap)), float(np.nanmax(heatmap))
    if hmax - hmin > 1e-8:
        norm = (heatmap - hmin) / (hmax - hmin)
    else:
        norm = np.zeros_like(heatmap)
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(norm)[:, :, :3]  # RGB
    heat_img = Image.fromarray((colored * 255).astype("uint8")).resize(original_img.size, Image.BILINEAR)
    blended = Image.blend(original_img.convert("RGBA"), heat_img.convert("RGBA"), alpha=alpha)
    return blended

# -------------------------
# Grad-CAM implementation (standard)
# -------------------------
def compute_gradcam(model: keras.Model, img_input: np.ndarray, conv_layer_name: str, 
                    class_index: Optional[int]=None, smooth_samples: int=0) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Standard Grad-CAM:
    - Find conv layer by name (recursive)
    - Compute gradients of target score w.r.t conv feature maps
    - Global-average-pool gradients -> weights -> weighted combination -> ReLU -> normalize
    If smooth_samples>0 applies SmoothGrad-like averaging with jitter.
    Returns heatmap (2D normalized) or (None, error_message).
    """
    # find conv layer
    conv_layer = find_layer_recursive(model, conv_layer_name)
    if conv_layer is None:
        return None, f"Conv layer '{conv_layer_name}' not found."

    # helper single run
    def _single_run(inp):
        # build model that outputs conv feature maps + preds
        try:
            grad_model = keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])
        except Exception as e:
            return None, f"Failed to build grad model: {e}"
        with tf.GradientTape() as tape:
            # ensure watch input
            tape.watch(inp)
            conv_outputs, preds = grad_model(inp)
            if class_index is None:
                # predicted class index
                # handle multi-dim outputs (binary classification with single logit/prob)
                try:
                    if preds.shape[-1] == 1:
                        score = preds[:, 0]
                    else:
                        # choose top predicted class
                        score = tf.reduce_max(preds, axis=-1)
                except Exception:
                    score = tf.reduce_mean(preds, axis=-1)
            else:
                # pick specific class index
                score = preds[:, class_index]
        grads = tape.gradient(score, conv_outputs)
        if grads is None:
            return None, "Gradients are None (disconnected graph)."
        # pooled weights
        if tf.rank(grads) == 4:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (channels,)
            conv_outputs = conv_outputs[0]  # (h,w,c)
            weights = pooled_grads
            # weighted sum
            cam = tf.tensordot(conv_outputs, weights, axes=([2], [0]))  # (h,w)
        else:
            return None, f"Unexpected grads rank: {tf.rank(grads)}"
        cam = tf.nn.relu(cam)
        cam_np = cam.numpy()
        # normalize
        if cam_np.max() - cam_np.min() > 1e-8:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np = cam_np * 0.0
        return cam_np, None

    # SmoothGrad-like averaging with small noise/jitter (optional)
    if smooth_samples is None or smooth_samples <= 0:
        cam, err = _single_run(img_input)
        return cam, err
    else:
        cams = []
        batch = img_input.copy()
        h, w = batch.shape[1], batch.shape[2]
        sigma = SMOOTH_SIGMA * max(h, w)
        for i in range(max(1, smooth_samples)):
            noisy = batch + np.random.normal(0, sigma, size=batch.shape).astype(np.float32)
            noisy = np.clip(noisy, -1.0, 1.0)  # keep in reasonable range for preprocess outputs
            cam_i, err = _single_run(noisy)
            if cam_i is None:
                return None, err
            cams.append(cam_i)
        cams = np.stack(cams, axis=0)
        cam_avg = np.mean(cams, axis=0)
        if cam_avg.max() - cam_avg.min() > 1e-8:
            cam_avg = (cam_avg - cam_avg.min()) / (cam_avg.max() - cam_avg.min())
        else:
            cam_avg = cam_avg * 0.0
        return cam_avg, None

# -------------------------
# Saliency fallback
# -------------------------
def compute_input_gradients_saliency(model: keras.Model, inp: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    img_var = tf.convert_to_tensor(inp)
    img_var = tf.cast(img_var, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_var)
        preds = model(img_var)
        try:
            score = preds[:, 0]
        except Exception:
            score = tf.reduce_mean(preds, axis=1)
    grads = tape.gradient(score, img_var)
    if grads is None:
        return None, "Input gradients are None."
    grads = tf.abs(grads)
    grads = tf.reduce_mean(grads, axis=-1)[0]  # (H,W)
    g = grads.numpy()
    if g.max() - g.min() > 1e-8:
        g = (g - g.min()) / (g.max() - g.min())
    else:
        g = g * 0.0
    return g, None

# -------------------------
# UI & main app flow
# -------------------------
# Load model
try:
    model, model_path = load_model_cached()
    st.sidebar.success(f"Loaded model: {model_path}")
except Exception as e:
    st.sidebar.error("Model load failed")
    st.error(str(e))
    st.stop()

# Candidate conv layers
candidates = list_conv_like_layers_recursive(model)
if not candidates:
    st.sidebar.error("No conv-like layers found in the model.")
    st.stop()

# Grad-CAM options
st.sidebar.markdown("### Grad-CAM options")
layer_choice = st.sidebar.selectbox("Choose conv layer", candidates, index=len(candidates)-1)
smooth_samples = st.sidebar.slider("SmoothGrad samples (0 = off)", min_value=0, max_value=32, value=SMOOTH_SAMPLES_DEFAULT, step=1)
alpha = st.sidebar.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
use_cmap = st.sidebar.selectbox("Colormap", ["jet", "inferno", "plasma", "magma"], index=0)

# File uploader
uploaded = st.file_uploader("Upload RBC image (jpg/png)", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an RBC image to run prediction & explanation.")
    st.stop()

# Load image and show
try:
    pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
except Exception as e:
    st.error("Failed to open image: " + str(e))
    st.stop()
st.image(pil_img, caption="Uploaded image", use_column_width=True)

# Preprocess
inp = preprocess_for_model(pil_img, model)

# Prediction
try:
    preds = model.predict(inp, verbose=0)
    # extract a scalar probability-like value for binary case
    try:
        prob = float(np.asarray(preds).ravel()[0])
    except Exception:
        prob = float(np.max(preds))
    st.markdown(f"<div class='result-card'><div style='font-size:1.6rem'>{'🔴 INFECTED' if prob>=0.5 else '🟢 HEALTHY'}</div><div style='color:#999'>{prob:.2%}</div></div>", unsafe_allow_html=True)
except Exception as e:
    st.error("Inference failed: " + str(e))
    st.stop()

# Compute Grad-CAM
debug_msgs = []
heatmap = None

try:
    cam, err = compute_gradcam(model, inp, layer_choice, class_index=None, smooth_samples=smooth_samples)
    if cam is None:
        debug_msgs.append(f"Grad-CAM error: {err}")
    else:
        heatmap = cam
        debug_msgs.append("Grad-CAM computed successfully.")
except Exception as e:
    debug_msgs.append("Grad-CAM runtime error: " + str(e))
    debug_msgs.append(traceback.format_exc())

# Fallback to saliency
if heatmap is None:
    try:
        sal, serr = compute_input_gradients_saliency(model, inp)
        if sal is None:
            debug_msgs.append("Saliency fallback failed: " + str(serr))
        else:
            heatmap = sal
            debug_msgs.append("Fell back to input-gradient saliency.")
    except Exception as e:
        debug_msgs.append("Saliency runtime error: " + str(e))
        debug_msgs.append(traceback.format_exc())

# Overlay & display
st.sidebar.text("\n".join(debug_msgs))
if heatmap is None:
    st.warning("Could not compute an explanation map.")
else:
    # resize/upsample heatmap to original image size
    heat_resized = tf.image.resize(heatmap[..., np.newaxis], size=pil_img.size[::-1], method="bilinear").numpy().squeeze()
    overlay = overlay_heatmap_pil(pil_img, heat_resized, alpha=alpha, cmap_name=use_cmap)
    st.image(overlay, caption="Grad-CAM / Saliency overlay", use_column_width=True)
    st.success("Explanation generated.")

st.markdown("---")
st.write("Tip: if Grad-CAM looks noisy, try a different conv layer or increase SmoothGrad samples.")
