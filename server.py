import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_URL = "books.csv"

st.title("Book Recommendation System")

@st.cache_data
def load_data(nrows = 15000):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.columns = data.columns.str.strip()
    data = data.rename(lowercase, axis='columns', inplace=False)
    data = data.dropna(axis='index', inplace=False)    
    data = data.reset_index(drop=False)
    print(data.info())
    return data
    

def combined_features(row):
    return str(row['rating'])+" "+str(row['genre'])+" "+str(row['author'])+" "+str(row['totalratings'])

def Cosine_Similarity(count_matrix):
    return cosine_similarity(count_matrix)

def get_index_from_title(data, title):
        return data[data.title == title]["index"].values[0]   

def get_title_from_index(data, index):
    return data[data.index == index]["title"].values[0]        


data_load_state = st.text('Loading data....')
data = load_data(15000)
data_load_state.text('Loading data....done!')
data["combined_features"] = data.apply(combined_features, axis =1)
cv = CountVectorizer()
count_matrix = cv.fit_transform(data["combined_features"])
cosine_sim = Cosine_Similarity(count_matrix)
print("Count Matrix:", count_matrix.toarray())
print(cosine_sim)
st.subheader('Raw data')
st.write(data)

book_user_likes = "Between Two Fires: American Indians in the Civil War"
print(data.columns)

# User Input 
user_input = st.text_input("Enter the book title that you like", "Harry Potter")

book_index = get_index_from_title(data, book_user_likes)
similar_books = list(enumerate(cosine_sim[book_index]))

sorted_similar_books = sorted(similar_books, key=lambda x:x[1], reverse=True)

i=0
for book in sorted_similar_books:
    st.write(get_title_from_index(data, int(book[0])))
    i=i+1
    if i>15:
        break