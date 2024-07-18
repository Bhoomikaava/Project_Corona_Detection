from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
import base64

app = Flask(__name__)

# Load the model
model = load_model('vgg-rps-final.h5')

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def predict_image(image_path):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Load and preprocess the image
    image = Image.open(image_path)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    
    # Run inference
    prediction = model.predict(data)
    y_classes = prediction.argmax(axis=-1)
    accu = prediction[0][y_classes] * 100
    
    # Determine the result
    if y_classes == 0:
        result = f"CoVid with accuracy: {accu}"
    elif y_classes == 1:
        result = f"Normal viridis with accuracy: {accu}"
    elif y_classes == 2:
        result = f"Pneumonia with accuracy: {accu}"
        
    # Convert image to base64 for displaying in HTML
    image_data = Image.fromarray(image_array)
    img_str = image_to_base64(image_data)
    
    return result, img_str

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    image_str = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            result, image_str = predict_image(file)
    return render_template('index.html', result=result, image=image_str)


@app.route('/pre', methods=['GET', 'POST'])
def pre():
    return render_template('pre.html')
if __name__ == '__main__':
    app.run(debug=True)
