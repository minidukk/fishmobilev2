from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Tải mô hình nhận diện cá cảnh
model = load_model('./mobilenetv2_MD.keras')

# Các lớp cá cảnh
classes = ['betta fish', 'corydoras fish', 'discus fish', 'flowerhorn fish', 
           'goldfish fish', 'guppy fish', 'neocaridina', 'neon fish', 
           'oscar fish', 'platy fish']

# Trang chủ với giao diện tải ảnh lên
@app.route('/')
def index():
    return render_template('index.html')

# API xử lý nhận diện hình ảnh
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files['file']
    image = Image.open(file).resize((224, 224))
    
    # Chuyển đổi hình ảnh thành mảng numpy
    image_array = np.array(image)
    
    # Tiền xử lý hình ảnh
    image_array = preprocess_input(image_array)
    
    # Thêm chiều cho batch
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    predicted_class = classes[np.argmax(predictions)]

    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
