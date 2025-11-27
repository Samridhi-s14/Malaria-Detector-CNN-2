# app.py — Robust Grad-CAM using head-model + dropdown + saliency fallback (fixed)
import io
import traceback
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.cm as cm

# -------------------------
# Page config + styling
# -------------------------
st.set_page_config(
    page_title="Malaria RBC Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalistic dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0a;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    h1 {
        color: #ffffff;
        font-weight: 300;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: 2px;
    }
    
    h2, h3 {
        color: #e0e0e0;
        font-weight: 300;
        font-size: 1.2rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .stFileUploader {
        background-color: #1a1a1a;
        border: 2px dashed #333333;
        border-radius: 12px;
        padding: 2rem;
    }
    
    .stFileUploader:hover {
        border-color: #4a4a4a;
    }
    
    .stFileUploader label {
        color: #cccccc !important;
        font-size: 0.95rem;
    }
    
    .result-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: 300;
        text-align: center;
        margin: 1rem 0;
        letter-spacing: 1px;
    }
    
    .infected {
        color: #ff4444;
    }
    
    .healthy {
        color: #44ff88;
    }
    
    .prob-display {
        text-align: center;
        color: #999999;
        font-size: 1rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #44ff88 0%, #ff4444 100%);
    }
    
    img {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        color: #cccccc;
    }
    
    p, span, label {
        color: #cccccc;
    }
    
    hr {
        border-color: #2a2a2a;
        margin: 3rem 0;
    }
</style>
""", unsafe_allow_html=True)


st.title("🩸 Malaria RBC Classifier — Grad-CAM (fixed)")

# -------------------------
# Config
# -------------------------
MODEL_PATHS = [
    "outputs/model_finetuned.h5",
    "model_finetuned.h5",
    "outputs/best_model.h5",
    "best_model.h5",
]
IMG_SIZE = (224, 224)

# -------------------------
# Helpers: load model, list conv layers, preprocessing
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

def list_candidate_conv_layers(model):
    """
    Collect candidate conv-like layers (4D outputs) from top-level and nested models.
    Returns list of (actual, actual, owner_name)
    where actual is either 'layer' or 'base/inner'.
    """
    seen = set()
    out = []
    for layer in model.layers:
        # top-level conv-like
        try:
            if hasattr(layer, "output_shape") and layer.output_shape is not None and len(layer.output_shape) == 4:
                actual = layer.name
                owner = "top"
                if actual not in seen:
                    out.append((actual, actual, owner))
                    seen.add(actual)
        except Exception:
            pass
        # nested model layers
        if isinstance(layer, tf.keras.Model):
            for sub in layer.layers:
                try:
                    if hasattr(sub, "output_shape") and sub.output_shape is not None and len(sub.output_shape) == 4:
                        actual = f"{layer.name}/{sub.name}"
                        owner = layer.name
                        if actual not in seen:
                            out.append((actual, actual, owner))
                            seen.add(actual)
                except Exception:
                    pass
    return out

def preprocess_pil(image: Image.Image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, 0)

def overlay_heatmap_pil(original_img: Image.Image, heatmap: np.ndarray, alpha=0.45):
    if heatmap is None:
        return None
    h = np.squeeze(heatmap)
    if h.ndim != 2:
        h = np.mean(h, axis=-1)
    h_min, h_max = float(h.min()), float(h.max())
    if h_max - h_min > 1e-8:
        u8 = np.uint8(255 * (h - h_min) / (h_max - h_min))
    else:
        u8 = np.uint8(h * 0)
    cmap = cm.get_cmap("jet")
    colored = cmap(u8)[:, :, :3]
    heat_img = Image.fromarray((colored * 255).astype("uint8")).resize(original_img.size, Image.BILINEAR)
    blended = Image.blend(original_img, heat_img, alpha=alpha)
    return blended

# -------------------------
# Grad-CAM core (head-model approach)
# -------------------------
def find_backbone(model, backbone_prefix="efficientnet"):
    # returns backbone model or None
    for L in model.layers:
        if isinstance(L, tf.keras.Model) and backbone_prefix in L.name.lower():
            return L
    # fallback: try any nested model
    for L in model.layers:
        if isinstance(L, tf.keras.Model):
            return L
    return None

def build_head_model_from_top(model, backbone):
    """
    Build a Keras Model that maps backbone.output shape -> final model output.
    We take the sequence of layers after the backbone in the top-level model and apply them to a new Input.
    """
    # find index of backbone in top-level model layers
    idx = None
    for i, L in enumerate(model.layers):
        if L is backbone or L.name == backbone.name:
            idx = i
            break
    if idx is None:
        raise RuntimeError("Could not locate backbone in model.layers")

    backbone_output_shape = backbone.output_shape  # e.g., (None, 7, 7, 1280) or (None, 1280)
    head_input_shape = backbone_output_shape[1:]
    head_input = keras.Input(shape=head_input_shape, name="head_input")
    x = head_input
    for L in model.layers[idx+1:]:
        try:
            x = L(x)
        except Exception as e:
            raise RuntimeError(f"Failed to apply layer '{L.name}' to head model input: {e}") from e
    head_model = keras.Model(inputs=head_input, outputs=x, name="head_model")
    return head_model

def make_gradcam_using_head(inp_array, model, backbone, conv_layer_name):
    """
    inp_array: preprocessed input batch (1,H,W,3)
    backbone: nested backbone model object
    conv_layer_name: name of conv layer inside backbone (e.g. 'top_conv' or nested name)
    """
    inner_conv_name = conv_layer_name.split("/",1)[-1]
    try:
        conv_layer = backbone.get_layer(inner_conv_name)
    except Exception as e:
        return None, f"Conv layer '{inner_conv_name}' not found inside backbone: {e}"

    intermediate_conv_model = keras.Model(inputs=backbone.input, outputs=conv_layer.output)

    try:
        head_model = build_head_model_from_top(model, backbone)
    except Exception as e:
        return None, f"Failed to build head model: {e}"

    conv_outputs = intermediate_conv_model(inp_array)  # shape (1,h,w,c) or (1,c)
    conv_outputs = tf.convert_to_tensor(conv_outputs)
    with tf.GradientTape() as tape:
        tape.watch(conv_outputs)
        preds = head_model(conv_outputs)
        # support different output shapes
        try:
            score = preds[:, 0]
        except Exception:
            score = tf.reduce_mean(preds, axis=1)
    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        return None, "gradients are None after wiring conv->head (disconnected)"
    # pooled grads and heatmap
    if tf.rank(grads) == 4:
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv0 = conv_outputs[0]  # (h,w,c)
        heatmap = conv0 @ pooled[..., tf.newaxis]
    elif tf.rank(grads) == 2:
        pooled = tf.reduce_mean(grads, axis=0)
        conv0 = conv_outputs[0]
        heatmap = conv0 * pooled
    else:
        return None, f"Unexpected conv_outputs rank: {tf.rank(conv_outputs)}"
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.size and heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap, None

# -------------------------
# Saliency fallback (input gradients)
# -------------------------
def compute_input_gradient_saliency(full_model, inp_array):
    img_var = tf.convert_to_tensor(inp_array)
    img_var = tf.cast(img_var, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_var)
        preds = full_model(img_var)
        try:
            score = preds[:, 0]
        except Exception:
            score = tf.reduce_mean(preds, axis=1)
    grads = tape.gradient(score, img_var)  # (1,H,W,3)
    if grads is None:
        return None, "input gradients is None"
    grads = tf.abs(grads)
    grads = tf.reduce_mean(grads, axis=-1)[0]  # (H,W)
    g = grads.numpy()
    if g.max() - g.min() > 1e-8:
        g = (g - g.min()) / (g.max() - g.min())
    else:
        g = np.zeros_like(g)
    return g, None

# -------------------------
# App logic
# -------------------------
# Load model
try:
    model, model_path = load_model_cached()
    st.sidebar.success(f"Loaded model: {model_path}")
except Exception as e:
    st.sidebar.error("Model load failed")
    st.error(str(e))
    st.stop()

# Determine backbone & candidate conv layers
backbone = find_backbone(model=model) if 'find_backbone' in globals() else None  # safe, but we'll override below

# Proper backbone lookup
backbone = find_backbone(model) if 'find_backbone' in globals() else None  # attempt to call if defined (rare)
# Overwrite with robust lookup
backbone = find_backbone(model) if callable(find_backbone) else None

# Simpler explicit lookup (guaranteed)
if backbone is None:
    backbone = find_backbone(model, backbone_prefix="efficientnet")
if backbone is None:
    # fallback: first nested model
    for L in model.layers:
        if isinstance(L, tf.keras.Model):
            backbone = L
            break

candidates_raw = list_candidate_conv_layers(model)
if not candidates_raw:
    st.sidebar.error("No conv-like layers found in model")
    st.stop()

# Build dropdown list
options = [f"{actual}  (owner={owner})" for (actual, actual, owner) in candidates_raw]
# select default: prefer something with top_conv
default_idx = 0
for i,(actual,_,owner) in enumerate(candidates_raw):
    if actual.endswith("top_conv") or actual.endswith("/top_conv") or actual=="top_conv":
        default_idx = i
        break

chosen = st.sidebar.selectbox("Select conv layer for Grad-CAM", options, index=default_idx)
conv_name = candidates_raw[options.index(chosen)][0]  # actual like 'efficientnetb0/top_conv' or 'top_conv'

# Upload
uploaded = st.file_uploader("Upload RBC image (jpg/png)", type=["jpg","jpeg","png"])
debug_box = st.sidebar.empty()

if not uploaded:
    st.info("Upload an RBC image to run prediction and explainability.")
else:
    try:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error("Failed to read uploaded image: " + str(e))
        st.stop()

    st.image(img, use_column_width=True)

    inp = preprocess_pil(img)
    try:
        pred = model.predict(inp, verbose=0)
        prob = float(np.asarray(pred).ravel()[0])
        st.markdown(f"<div class='result-card'><div class='prediction-text'>{'🔴 INFECTED' if prob>=0.5 else '🟢 HEALTHY'}</div>"
                    f"<div class='prob'>confidence: {prob:.2%}</div></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error("Inference failed: " + str(e))
        st.sidebar.text(traceback.format_exc())
        st.stop()

    # Attempt Grad-CAM using head-model approach when possible
    debug_lines = []
    heatmap = None

    # Build head-model Grad-CAM if backbone exists
    if backbone is not None:
        try:
            heatmap, err = make_gradcam_using_head(inp, model, backbone, conv_name)
            if heatmap is not None:
                debug_lines.append("Grad-CAM via head-model succeeded (backbone path).")
        except Exception as e:
            debug_lines.append("Head-model Grad-CAM failed: " + str(e))
            debug_lines.append(traceback.format_exc())

    # If not successful, fallback to base-input intermediate attempt for nested convs
    if heatmap is None:
        try:
            if "/" in conv_name:
                base_name, nested_name = conv_name.split("/",1)
                base_model = None
                for L in model.layers:
                    if isinstance(L, tf.keras.Model) and L.name == base_name:
                        base_model = L; break
                if base_model is not None:
                    try:
                        conv_layer = base_model.get_layer(nested_name)
                        inter = keras.Model(inputs=base_model.input, outputs=conv_layer.output)
                        debug_lines.append("Built intermediate from base.input -> conv.output")
                        with tf.GradientTape() as tape:
                            conv_outputs = inter(inp)
                            preds = model(inp)
                            loss = preds[:,0]
                        grads = tape.gradient(loss, conv_outputs)
                        if grads is not None:
                            pooled = tf.reduce_mean(grads, axis=(0,1,2))
                            conv0 = conv_outputs[0]
                            h = conv0 @ pooled[..., tf.newaxis]
                            h = tf.squeeze(h).numpy()
                            h = np.maximum(h,0)
                            if h.max()>0:
                                h = h / h.max()
                            heatmap = h
                            debug_lines.append("Alternate intermediate gradcam succeeded")
                        else:
                            debug_lines.append("Alternate intermediate yields None grads")
                    except Exception as e:
                        debug_lines.append("Alternate intermediate build error: " + str(e))
                else:
                    debug_lines.append("Base model for nested conv not found")
            else:
                debug_lines.append("No nested conv naming; skipping base-input fallback")
        except Exception as e:
            debug_lines.append("Fallback intermediate runtime error: " + str(e))
            debug_lines.append(traceback.format_exc())

    # If still None, compute saliency map
    if heatmap is None:
        try:
            sal, serr = compute_input_gradient_saliency(model, inp)
            if sal is not None:
                heatmap = sal
                debug_lines.append("Fell back to input-gradient saliency (reliable).")
            else:
                debug_lines.append("Saliency fallback failed: " + str(serr))
        except Exception as e:
            debug_lines.append("Saliency runtime error: " + str(e))
            debug_lines.append(traceback.format_exc())

    # Display debug and overlay
    debug_box.text("\n".join(debug_lines) if debug_lines else "No debug messages.")
    if heatmap is None:
        st.warning("Could not compute an attention map.")
    else:
        overlay = overlay_heatmap_pil(img, heatmap, alpha=0.45)
        st.image(overlay, caption="Explanation overlay", use_column_width=True)
        st.success("Explanation generated (Grad-CAM or saliency).")

st.markdown("---")
st.write("Tip: if Grad-CAM looks noisy, try a different conv layer from the sidebar dropdown.")
