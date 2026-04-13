from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import custom_object_scope
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Patch for Keras version mismatch: model was saved with a version that
# includes 'quantization_config' in Dense, which older versions don't recognise.
class _PatchedDense(Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

with custom_object_scope({'Dense': _PatchedDense}):
    model = load_model("blood_group_model_2.h5")

labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

def preprocess_image(image):
    image = image.resize((128,128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert('RGB')

    processed = preprocess_image(image)
    prediction = model.predict(processed)

    predicted_class = labels[np.argmax(prediction)]

    return jsonify({
        "prediction": predicted_class
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    