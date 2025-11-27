import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.cm as cm
import io

# ------------------------------
# Load model
# ------------------------------
MODEL_PATH = "outputs/model_finetuned.h5"

@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

IMG_SIZE = (224, 224)

# ------------------------------
# Grad-CAM Utilities (NO OpenCV)
# ------------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    return heatmap


def overlay_heatmap_pil(original_img, heatmap, alpha=0.45):
    """Overlay heatmap on image without cv2 (PIL + matplotlib only)."""
    heatmap = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap)[:, :, :3]

    heatmap_img = Image.fromarray((colored_heatmap * 255).astype("uint8"))
    heatmap_img = heatmap_img.resize(original_img.size, Image.BILINEAR)

    blended = Image.blend(original_img, heatmap_img, alpha=alpha)
    return blended


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if "conv" in layer.name or len(layer.output_shape) == 4:
            return layer.name
    return None


LAST_CONV_LAYER = find_last_conv_layer(model)

# ------------------------------
# Preprocessing
# ------------------------------

def preprocess_pil_image(image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, 0)


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Malaria RBC Classifier", layout="centered")

st.title("🩸 Malaria RBC Classification Dashboard")
st.write("Upload a red blood cell microscope image to detect malaria infection.")

uploaded_file = st.file_uploader("Upload RBC Image", type=["jpg","jpeg","png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, width=300)

    # Preprocess
    img_array = preprocess_pil_image(image)

    # Predict
    prob = float(model.predict(img_array)[0][0])
    label = "🟢 Healthy" if prob < 0.5 else "🔴 Infected"

    st.subheader("Prediction")
    st.write(f"**Result:** {label}")

    st.progress(min(max(prob if prob>=0.5 else 1-prob, 0.01), 0.99))

    st.write(f"**Probability:** `{prob:.4f}`")

    # Grad-CAM
    st.subheader("🔍 Grad-CAM Heatmap")
    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
    heatmap_overlay = overlay_heatmap_pil(image, heatmap)

    st.image(heatmap_overlay, caption="Grad-CAM Visualization", use_column_width=True)

    st.success("Inference + Grad-CAM completed successfully!")

else:
    st.info("Please upload an RBC image to begin.")
