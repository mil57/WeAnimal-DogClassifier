---
topic: WeAnimal-DogClassifier
languages: python
Platform: Heroku
Framwork: Flask
---

This is a simple ML classifier for breeds of dog

# How to use
https://wedog.herokuapp.com/ 
check the status of the app. Returns "Hiiiiii Woooorld!"

https://wedog.herokuapp.com/<URL to a picture of a dog>
Classify the dog in the picture. Return a string list of 5 most likely breeds: ["Probability\tBreed",...] 
  Example:
  GET https://wedog.herokuapp.com/https://cdn1.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg
  Response: ["99.93%\tgolden_retriever", "0.04%\tBrittany_spaniel", "0.02%\tLabrador_retriever", "0.01%\tLeonberg", "0.00%\tChesapeake_Bay_retriever"]
