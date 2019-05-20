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


from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"
