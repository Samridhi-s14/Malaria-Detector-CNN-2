
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('outputs/model_finetuned.h5')

def preprocess_image_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224,224))
    arr = np.array(img).astype('float32')
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if file is None:
        return jsonify({'error': 'no image received'}), 400
    img = preprocess_image_bytes(file.read())
    prob = float(model.predict(img)[0,0])
    label = 'infected' if prob >= 0.5 else 'healthy'
    return jsonify({'label': label, 'probability': prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
