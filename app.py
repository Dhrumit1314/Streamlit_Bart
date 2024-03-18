# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:34:09 2024

@author: Dhrumit Patel
"""

# # BART Model
# from flask import Flask, render_template, request
# from transformers import BartTokenizer, TFBartForConditionalGeneration
# import tensorflow as tf

# app = Flask(__name__)

# # Load the BART model and tokenizer
# tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large')

# with tf.device('/CPU:0'):
#     model_bart = TFBartForConditionalGeneration.from_pretrained('models/BART_Pretrained_Model')

# def summarize_text_bart(text):
#     # Preprocess text
#     inputs = tokenizer_bart([text], max_length=1024, return_tensors='tf')

#     with tf.device('/CPU:0'):
#         # Perform text summarization
#         summary_ids = model_bart.generate(inputs['input_ids'], num_beams=4, max_length=256, early_stopping=True)

#     # Decode and print the summary
#     output = [tokenizer_bart.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

#     return output[0]

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         text = request.form['text']
#         summary = summarize_text_bart(text)
#         return render_template('index.html', summary=summary)
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


import streamlit as st
from transformers import BartTokenizer, TFBartForConditionalGeneration
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def get_model():
    # Specify the device
    with tf.device('/CPU:0'):
        # Load pre-trained model and tokenizer
        tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large')
        model_bart = TFBartForConditionalGeneration.from_pretrained('Dhrumit1314/BART_TextSummary')
    return tokenizer_bart, model_bart

tokenizer_bart, model_bart = get_model()

def summarize_text_bart(text):
    # Preprocess text
    inputs = tokenizer_bart([text], max_length=1024, return_tensors='tf')

    # Perform text summarization
    with tf.device('/CPU:0'):
        summary_ids = model_bart.generate(inputs['input_ids'], num_beams=4, max_length=256, early_stopping=True)

    # Decode and print the summary
    output = [tokenizer_bart.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

    return output[0]

user_input = st.text_area('Enter Text to Summarize')
button = st.button("Summarize")

if user_input and button:
    summary = summarize_text_bart(user_input)
    st.write("Summary: ", summary, height=200)
