''' This code is a first attempt at using the streamlit library to build an (I think) website. I'm also using the iris data set from sklearn the random forest classifier from sklearn. The idea is to build a small interactive module and give a prediction of what species of flower has the particular attributes of the user input. It is almost exactly the same as the code written by Krish Naik in his Udemy course. '''



import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Cache the data loading function
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names


# Load data and target names
df, target_names = load_data()

# Display target names (for debugging)
#st.write(f"Target Names: {target_names}")
#st.write(df.head())

# Cache model fitting
@st.cache_data
def train_model():
    model = RandomForestClassifier()
    model.fit(df.iloc[:, :-1], df['species'])
    return model

# Train the model
model = train_model()

# Sidebar for user input
st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()), float(df["petal length (cm)"].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()), float(df["petal width (cm)"].mean()))

user_input = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make prediction
prediction = model.predict(user_input)
predicted_species = target_names[prediction[0]]

# Display prediction result
st.write(f"Predicted Species: **{predicted_species}**")
