# Machine Learning Stuff
import keras
import tensorflow as tf
from keras.models import model_from_json
import json
import os
from PIL import Image
from skimage.io import imread
from keras.applications.densenet import DenseNet121, preprocess_input
import numpy as np
import requests
from io import BytesIO


# Init machine learning model
with open('labels',"r") as json_file_labels:
    labels = json.loads(json_file_labels.read().replace("\'","\""))

with open('model.json') as json_file:
    json_string = json.load(json_file)
    model = model_from_json(json_string)

model.load_weights('dog_breed_classifier_model.h5')
model._make_predict_function()


def download_and_predict(url, filename):
    # download and save
    response = requests.get(url)
    
    print("Got img!************* \n")
    
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    img = img.resize((224, 224))

    print("PIL************* \n")
    
    # predict
    img = np.array(img)
    print("conversion!************* \n")
    img = preprocess_input(img)
    print("preprocessed!************* \n")
    probs = model.predict(np.expand_dims(img, axis=0))
    
    print("Predicted!************* \n")
    
    result = []
    for idx in probs.argsort()[0][::-1][:5]:
        result.append("{:.2f}%".format(probs[0][idx]*100) + "\t" + labels[idx].split("-")[-1])
        
    print("About to return!************* \n")
    
    # cLEAN UP AND RETURN
    return json.dumps(result)


# app stuff
from flask import Flask
app = Flask(__name__)


#port = int(os.environ.get('PORT', 5000))
#app.run(host='0.0.0.0', port=port)


# Index, and test method
@app.route("/")
def hello():
    return "hiiiiiiiiii wooooooolrd"


# Main classifier service, accpet a URL to the picute as GET parameter
@app.route('/<path:subpath>')
def classify(subpath):
    return download_and_predict(subpath, "test.jpg")
