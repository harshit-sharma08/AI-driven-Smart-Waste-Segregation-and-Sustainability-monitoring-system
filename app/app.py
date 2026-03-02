import numpy as np
import tensorflow as tf
from flask import Flask,request,jsonify
from PIL import Image
from io import BytesIO
app = Flask(__name__)
model = tf.keras.models.load_model('model/waste_classifier_model.keras')
class_names = ['metal','organic','paper','plastic']
def preprocess_image(image):
    image = Image.convert('RGB')
    image = image.resize((224,224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array,axis = 0)
    return img_array
@app.route("/predict",methods = ['POST'])
def predict():
    if request.data == b'':
        return jsonify({"error":"No image Received"})
    image = Image.open(BytesIO(request.data))
    image_array = preprocess_image(image)
    preds = model.predict(image_array)
    predicted_index = np.argmax(preds)
    predicted = class_names[predicted_index]
    confidence = np.max(preds)*100
    return jsonify({"prediction":predicted,
                    "confidence":round(confidence,2)})
if __name__ == "__main__":
    app.run(debug = True)

