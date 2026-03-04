import streamlit as st 
import pickle
import pandas as pd
st.title("Movies Recommended Application")
list_mov=pickle.load(open("movies_list.pkl","rb"))
selected_movie_name= st.selectbox(" Enter your Movies Name ",options=[v for i,v in list_mov.items()])
movies=pickle.load(open("movies.pkl","rb"))
# print(movies)
similarity= pickle.load(open("similarity.pkl","rb"))
def recommend(movie):
    movie_index= movies[movies["title"]==movie].index[0]
    distances=similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)), reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies=[]
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies


if st.button("Recommend"):
    recommendations= recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)