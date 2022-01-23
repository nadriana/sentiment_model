# Creating REST API using FAST API

import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

# input form
app = FastAPI()

# esto es un ejemplo. No es parte del modelo
#@app.get('/', response_class=HTMLResponse)
#def root():
#    return f"<span> hello </span>"
# ---

@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}


@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="Text Emotion to be tested" />
        <input type="submit" />'''

# We have created our FastAPI app in the first line and used the get method on the /predict route, 
# which will return an HTML response so that the user can see a real HTML page,
#  and input the data on forms using the post method.
#  We will use that data to predict on.

# esto se corre en terminal. 
# El path tiene que ser el correcto, ie, donde están los archivos model.py y app.py
# yo lo corrí en la terminal integrada en VSC porque ya tiene el path correcto
# uvicorn app:app --reload


# are essentially doing the same work for cleaning and preprocessing data,
# which we have used in our model.py file.
data = pd.read_csv('archive/Sentiment.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)

def preProcess_data(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

def my_pipeline(text):
    text_new = preProcess_data(text)
    X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    X = pad_sequences(X, maxlen=28)
    return X

# POST request
# create a POST request at the "/predict" route so that the data posted 
# using the form can be passed into our model, 
# and we can make predictions.
@app.post('/predict')
def predict(text:str = Form(...)):
    clean_text = my_pipeline(text) #clean, and preprocess the text through pipeline
    loaded_model = tf.keras.models.load_model('sentiment.h5') #load the saved model 
    predictions = loaded_model.predict(clean_text) #predict the text
    sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
    probability = max(predictions.tolist()[0]) #calulate the probability
    if sentiment==0:
         t_sentiment = 'negative' #set appropriate sentiment
    elif sentiment==1:
         t_sentiment = 'neutral'
    elif sentiment==2:
         t_sentiment='postive'
    return { #return the dictionary for endpoint
         "ACTUALL SENTENCE": text,
         "PREDICTED SENTIMENT": t_sentiment,
         "Probability": probability
    }

# para parar usar control C (control no cmnd)

#You can add a .gitignore file to ignore the files which you will not use:
#en terminal: touch .gitignore
# ahí se agrega
# __pycache__ #para que no suba esto
# model.py #para que no suba el modelo ya que no es necesario, pero se puede dejar