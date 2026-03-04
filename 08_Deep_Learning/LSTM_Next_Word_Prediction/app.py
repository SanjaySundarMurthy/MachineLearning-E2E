# create a streamlit web app with functions importing model
import streamlit as st
# import the above model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# load the dataset
with open('C:\\Users\\ssan\\Downloads\\Cursor_Projects\\Deep_Learning\\LSTMRNN\\hamlet.txt','r') as file:
  texts=file.read().lower()

tokenizer=Tokenizer()
tokenizer.fit_on_texts([texts])
total_words=len(tokenizer.word_index)+1
# print(total_words)


# load the model
model=load_model('C:\\Users\\ssan\\Downloads\\Cursor_Projects\\Deep_Learning\\LSTMRNN\\lstmrnn.keras')

# title of application
st.title("Next Word Prediction")
text=st.text_input("Enter the text")
num_of_words=st.slider("Number of words",min_value=1,max_value=10)
# create a function for prediction 
def predict_next_word(text,num_of_words):
    for _ in range(num_of_words):
        token_list=tokenizer.texts_to_sequences([text])[0]
        padded_list=np.array(pad_sequences([token_list],maxlen=13,padding='pre'))
        prediction=np.argmax(model.predict(padded_list,verbose=0),axis=1)
        output_word=""
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
