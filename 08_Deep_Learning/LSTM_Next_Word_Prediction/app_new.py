# create a streamlit web app with functions importing model
import streamlit as st
# import the above model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# load the model
model=load_model('/content/lstmrnn.h5')

# title of application
st.title("Next Word Prediction")
text=st.text_input("Enter the text")
num_of_words=st.slider("Number of words",min_value=1,max_value=10)
# create a function for prediction 
def predict_next_word(text,num_of_words):
  for _ in range(len(num_of_words)):
    token_list=tokenizer.texts_to_sequences([text])[0]
    max_sequence=max([len(x) for x in token_list ])
    token_list=np.array(pad_sequences([token_list],maxlen=max_sequence-1,padding='pre'))
    prediction=np.argmax(model.predict(token_list,verbose=0),axis=1)
    output_words=""
    for word, index in tokenizer.word_index.items():
      if index==prediction:
        output_word=word
        break
    text+=" "+output_word
    return text
# create a button
button=st.button("Generate Words")
if (button):
  output=predict_next_word(text,num_of_words)
  st.write(output)
