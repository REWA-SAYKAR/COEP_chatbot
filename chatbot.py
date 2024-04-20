import json
import random
import pickle
import numpy as np
from colorama import Fore,init
init()
grn = Fore.GREEN
blu = Fore. BLUE

import logging
import os

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intentsplus.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    p = bow(sentence, words,show_details=False)
    if p is None:
        return []

    res = model.predict(np.array([p]),verbose=0)[0]
    
    ERROR_THRESHOLD = 0.15
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = ["Please Try again with more specifics in your sentence"]
    for i in list_of_intents:
        i['tag'] = i['tag'].replace(" " , "_")
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res


while True:
    print(grn+'')
    message = input("You: ")

    if not message:
        print(blu+'Bot: Please enter a message')
        continue

    if message.lower() in ['quit','exit','close']:
        break
    elif  any(str(a) + symbol + str(b) in message for a in [1,2,3,4,5,6,7,8,9,0] for b in [1,2,3,4,5,6,7,8,9,0] for symbol in ['*','/','-','+','**']):
        index = []
        for i in range(len(message)):
            if message[i].isnumeric():
                index.append(i)
        try:
            ans = eval(message[index[0]:index[-1]+1])
        except Exception as e:
            print(blu+'Bot: Error ! ',e)
        else:
            print(blu+'Bot: Answer is ',ans)
    else:
        if message.lower() in ['hi','hello','hey']:
            print(blu+"Bot: Hello, How can I help you?")
            continue

        try:
            # for i in range(3):
                # response+=chatbot_response(message)
            
            response=""
            response+=chatbot_response(message)

            print(blu+"Bot:", response)
        except Exception as e:
            print(blu+'Bot: Invalid Question! ',"Please ask me questions related to the COEP.")
            continue