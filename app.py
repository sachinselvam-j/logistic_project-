import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Titanic Survival Prediction 🚢")

# Load dataset
@st.cache_data
def load_and_train():
    df = pd.read_csv("Titanic-Dataset.csv")

    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])  # male=1, female=0

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model

model = load_and_train()

st.success("Model trained successfully ✅")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", 0.0, 100.0, 25.0)
sibsp = st.number_input("Siblings / Spouse", 0, 10, 0)
parch = st.number_input("Parents / Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)

sex = 1 if sex == "Male" else 0

if st.button("Predict"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
