import io
import os
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify

import model

# def read_image():
app = Flask(__name__)

@app.route('/ping', methods=["GET"])
def health():
    return jsonify({'msg':"pong"})

@app.route('/v1/predict', methods=['POST'])
def predict_torch():
    file = request.files['file']
    img = file.read()
    img = Image.open(io.BytesIO(img))
    img = img.convert("RGB") 
    prediction = model.predict_torch(img)
    resultJson = jsonify(prediction)
    return resultJson

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)