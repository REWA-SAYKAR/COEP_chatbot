import streamlit as st
import replicate
import os
from textblob import TextBlob
from datetime import datetime

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
    try:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        result = ["Please Try again with more specifics in your sentence"]
        for i in list_of_intents:
            i['tag'] = i['tag'].replace(" " , "_")
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result
    except Exception as e:
        return "Please Try again with more specifics in your sentence"

def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res


st.set_page_config(page_title="DSci: Chatbot")

with st.sidebar:
    st.title('DSci Lab Project: Chatbot')

    st.success("Let's Go!", icon='✌️')

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

if "interaction_count" not in st.session_state.keys():
    st.session_state.interaction_count = 1

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def get_sentiment(message):
    analysis = TextBlob(message)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]
    st.session_state.interaction_count = 1

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
st.sidebar.subheader(f"Interactions: {st.session_state.interaction_count}")

def generate_llama2_response(prompt_input):
    response = chatbot_response(prompt_input)
    return response

def process_user_input(prompt):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment = get_sentiment(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp, "sentiment": sentiment})
    
    with st.chat_message("user"):
        st.write(f"{timestamp} - User ({sentiment}): {prompt}")

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
        st.session_state.interaction_count += 1

if prompt := st.text_input("You: ", key="input"):
    process_user_input(prompt)