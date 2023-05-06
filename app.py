import streamlit as st
import numpy as np
from PIL import Image
from pickle import load

scaler = load(open('C:/Users/user/Desktop/Internship Data Science 2023/titanic_survival_prediction/models/standard_scaler.pkl', 'rb'))
lr_model = load(open('C:/Users/user/Desktop/Internship Data Science 2023/titanic_survival_prediction/models/lr_model.pkl', 'rb'))

st.header("Titanic Survival Prediction")

st.write("The sinking of the Titanic is one of the most infamous shipwrecks in history.")

st.write("On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.")

st.write("While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.")

st.write("I am here to predict whether a person survived or not based on the following information.")

st.write("If the resulting value is 1, it means that had survived and if the resulting value is 0, it means that the person had died.")

img = Image.open("C:/Users/user/Desktop/Internship Data Science 2023/titanic_survival_prediction/titanic.png")
st.image(img)

Age = st.text_input("Age", placeholder="Enter Age")
Fare = st.text_input("Fare", placeholder="Enter Fare")
Pclass_1 = st.text_input("Pclass_1", placeholder="Enter 1(if the person belongs to class 1) otherwise 0")
Pclass_2 = st.text_input("Pclass_2", placeholder="Enter 1(if the person belongs to class 2) otherwise 0")
Pclass_3 = st.text_input("Pclass_3", placeholder="Enter 1(if the person belongs to class 3) otherwise 0")
Sex_female = st.text_input("Sex_female", placeholder="Enter 1(if the person is female) otherwise 0")
Sex_male = st.text_input("Sex_male", placeholder="Enter 1(if the person is male) otherwise 0")
SibSp_0 = st.text_input("SibSp_0", placeholder="Enter 1(if the person belongs to SibSp_0) otherwise 0")
SibSp_1 = st.text_input("SibSp_1", placeholder="Enter 1(if the person belongs to SibSp_1) otherwise 0")
SibSp_2 = st.text_input("SibSp_2", placeholder="Enter 1(if the person belongs to SibSp_2) otherwise 0")
SibSp_3 = st.text_input("SibSp_3", placeholder="Enter 1(if the person belongs to SibSp_3) otherwise 0")
SibSp_4 = st.text_input("SibSp_4", placeholder="Enter 1(if the person belongs to SibSp_4) otherwise 0")
SibSp_5 = st.text_input("SibSp_5", placeholder="Enter 1(if the person belongs to SibSp_5) otherwise 0")
SibSp_8 = st.text_input("SibSp_8", placeholder="Enter 1(if the person belongs to SibSp_8) otherwise 0")
Parch_0 = st.text_input("Parch_0", placeholder="Enter 1(if the person belongs to Parch_0) otherwise 0")
Parch_1 = st.text_input("Parch_1", placeholder="Enter 1(if the person belongs to Parch_1) otherwise 0")
Parch_2 = st.text_input("Parch_2", placeholder="Enter 1(if the person belongs to Parch_2) otherwise 0")
Parch_3 = st.text_input("Parch_3", placeholder="Enter 1(if the person belongs to Parch_3) otherwise 0")
Parch_4 = st.text_input("Parch_4", placeholder="Enter 1(if the person belongs to Parch_4) otherwise 0")
Parch_5 = st.text_input("Parch_5", placeholder="Enter 1(if the person belongs to Parch_5) otherwise 0")
Parch_6 = st.text_input("Parch_6", placeholder="Enter 1(if the person belongs to Parch_6) otherwise 0")
Embarked_C = st.text_input("Embarked_C", placeholder="Enter 1(if the person belongs to Embarked_C) otherwise 0")
Embarked_Q = st.text_input("Embarked_Q", placeholder="Enter 1(if the person belongs to Embarked_Q) otherwise 0")
Embarked_S = st.text_input("Embarked_S", placeholder="Enter 1(if the person belongs to Embarked_S) otherwise 0")

btn = st.button("Predict")

if btn == True:
    if Age and Fare and Pclass_1 and Pclass_2 and Pclass_3 and Sex_female and Sex_male and SibSp_0 and SibSp_1 and SibSp_2 and SibSp_3 and SibSp_4 and SibSp_5 and SibSp_8 and Parch_0 and Parch_1 and Parch_2 and Parch_3 and Parch_4 and Parch_5 and Parch_6 and Embarked_C and Embarked_Q and Embarked_S:
        query_point = np.array([float(Age), float(Fare), float(Pclass_1), float(Pclass_2), float(Pclass_3), float(Sex_female), float(Sex_male), float(SibSp_0), float(SibSp_1), float(SibSp_2), float(SibSp_3), float(SibSp_4), float(SibSp_5), float(SibSp_8), float(Parch_0), float(Parch_1), float(Parch_2), float(Parch_3), float(Parch_4), float(Parch_5), float(Parch_6), float(Embarked_C), float(Embarked_Q), float(Embarked_S)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = lr_model.predict(query_point_transformed)
        st.success(pred)
    else:
        st.error("Enter the values properly.")