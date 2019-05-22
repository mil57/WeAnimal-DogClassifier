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

label = ['n02105162-malinois', 'n02094258-Norwich_terrier', 'n02102177-Welsh_springer_spaniel', 'n02086646-Blenheim_spaniel', 'n02086910-papillon', 'n02093256-Staffordshire_bullterrier', 'n02113624-toy_poodle', 'n02105056-groenendael', 'n02109961-Eskimo_dog', 'n02116738-African_hunting_dog', 'n02096177-cairn', 'n02096585-Boston_bull', 'n02100735-English_setter', 'n02102973-Irish_water_spaniel', 'n02099429-curly-coated_retriever', 'n02088364-beagle', 'n02101006-Gordon_setter', 'n02108089-boxer', 'n02097130-giant_schnauzer', 'n02112137-chow', 'n02107574-Greater_Swiss_Mountain_dog', 'n02113186-Cardigan', 'n02092339-Weimaraner', 'n02092002-Scottish_deerhound', 'n02107312-miniature_pinscher', 'n02095570-Lakeland_terrier', 'n02100877-Irish_setter', 'n02109047-Great_Dane', 'n02093991-Irish_terrier', 'n02102480-Sussex_spaniel', 'n02093859-Kerry_blue_terrier', 'n02108915-French_bulldog', 'n02110958-pug', 'n02105505-komondor', 'n02085936-Maltese_dog', 'n02105412-kelpie', 'n02098413-Lhasa', 'n02088466-bloodhound', 'n02091032-Italian_greyhound', 'n02091134-whippet', 'n02091244-Ibizan_hound', 'n02113712-miniature_poodle', 'n02096051-Airedale', 'n02100236-German_short-haired_pointer', 'n02091831-Saluki', 'n02097298-Scotch_terrier', 'n02112706-Brabancon_griffon', 'n02113799-standard_poodle', 'n02090622-borzoi', 'n02087394-Rhodesian_ridgeback', 'n02108551-Tibetan_mastiff', 'n02094433-Yorkshire_terrier', 'n02093754-Border_terrier', 'n02113023-Pembroke', 'n02087046-toy_terrier', 'n02105641-Old_English_sheepdog', 'n02099849-Chesapeake_Bay_retriever', 'n02089973-English_foxhound', 'n02088238-basset', 'n02099601-golden_retriever', 'n02109525-Saint_Bernard', 'n02101556-clumber', 'n02098286-West_Highland_white_terrier', 'n02097474-Tibetan_terrier', 'n02099267-flat-coated_retriever', 'n02111500-Great_Pyrenees', 'n02106550-Rottweiler', 'n02085620-Chihuahua', 'n02089867-Walker_hound', 'n02106382-Bouvier_des_Flandres', 'n02097658-silky_terrier', 'n02110806-basenji', 'n02095314-wire-haired_fox_terrier', 'n02096437-Dandie_Dinmont', 'n02091467-Norwegian_elkhound', 'n02104365-schipperke', 'n02091635-otterhound', 'n02107683-Bernese_mountain_dog', 'n02093428-American_Staffordshire_terrier', 'n02102318-cocker_spaniel', 'n02111889-Samoyed', 'n02110185-Siberian_husky', 'n02107908-Appenzeller', 'n02085782-Japanese_spaniel', 'n02107142-Doberman', 'n02090721-Irish_wolfhound', 'n02104029-kuvasz', 'n02110063-malamute', 'n02106030-collie', 'n02093647-Bedlington_terrier', 'n02088632-bluetick', 'n02110627-affenpinscher', 'n02101388-Brittany_spaniel', 'n02108000-EntleBucher', 'n02106166-Border_collie', 'n02097047-miniature_schnauzer', 'n02086240-Shih-Tzu', 'n02105251-briard', 'n02111129-Leonberg', 'n02113978-Mexican_hairless', 'n02112350-keeshond', 'n02097209-standard_schnauzer', 'n02111277-Newfoundland', 'n02112018-Pomeranian', 'n02098105-soft-coated_wheaten_terrier', 'n02086079-Pekinese', 'n02105855-Shetland_sheepdog', 'n02089078-black-and-tan_coonhound', 'n02095889-Sealyham_terrier', 'n02094114-Norfolk_terrier', 'n02096294-Australian_terrier', 'n02099712-Labrador_retriever', 'n02100583-vizsla', 'n02115913-dhole', 'n02090379-redbone', 'n02106662-German_shepherd', 'n02102040-English_springer', 'n02108422-bull_mastiff', 'n02088094-Afghan_hound', 'n02115641-dingo']
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
        result.append("{:.2f}%".format(probs[0][idx]*100) + "\t" + label[idx].split("-")[-1])
        
    print("About to return!************* \n")
    
    # cLEAN UP AND RETURN
    return json.dumps(result)


# app stuff
from flask import Flask
app = Flask(__name__)



# Index, and test method
@app.route("/")
def hello():
    return "hiiiiiiiiii wooooooolrd"


# Main classifier service, accpet a URL to the picute as GET parameter
@app.route('/<path:subpath>')
def classify(subpath):
    return download_and_predict(subpath, "test.jpg")

