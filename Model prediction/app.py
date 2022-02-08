from re import X
from flask import Flask,render_template,url_for,request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models 
import numpy as np
import pickle

french_tokenizer = pickle.load(open('french_tokenizer.pickle', 'rb'))
english_tokenizer = pickle.load(open('english_tokenizer.pickle', 'rb'))
model = models.load_model("translator_model.h5")

y_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
y_id_to_word[0] = '<PAD>'
#y_id_to_word

app = Flask(__name__)

@app.route('/')
def hello_World():
	return "Hello Soumya"

@app.route('/translator', methods = ['GET', 'POST'])
def eng_to_french():
    message = request.args.get("message")
    sentence = [english_tokenizer.word_index[word] for word in message.split()]
    #sentence
    sentence = pad_sequences([sentence], maxlen=15, padding='post')
    sentences = np.array([sentence[0]])
    predictions = model.predict(sentences, len(sentences))
    x = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
    if '<PAD>' in x:
        x=x.replace('<PAD>','')
    print(x)    
    return x

if __name__ == '__main__':
	app.run(debug=True)
	