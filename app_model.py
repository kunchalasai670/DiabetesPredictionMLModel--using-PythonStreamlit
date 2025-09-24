#----------------- Import Libraries (Start)---------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#----------------- Import Libraries (End)-----------------------------------------------------------------------------------------------------

#----------------- Load Data in Dataframe (Start)---------------------------------------------------------------------------------------------

df  = pd.read_csv('H:\\AV_DEVS_Projects\\Python Projects\\DiabetesPredictionMLModel\\pima_diabetes.csv')

#----------------- Load Data in Dataframe (End)-----------------------------------------------------------------------------------------------

#----------------- Streamlit Web App (Start)--------------------------------------------------------------------------------------------------

# Main header of the app:
st.title('Diabetes Checkup')

# Header to show the description of training data:
st.subheader('Training Data')
st.write(df.describe())

st.subheader('Visualization')
st.bar_chart(df)

# Segrigating data into independant variable (X) and dependant varible (y):
X = df.drop(['Outcome'], axis = 1) # dropping the last "Outcome" column
y = df.iloc[:, -1] # Storing only the Outcome column

# Spliting the data into training and test datasets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)


def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp= st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi =  st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report, index = [0])
    return report_data

user_data = user_report()
print(user_data)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(X_test))*100)+'%')

user_result = rf.predict(user_data)
st.subheader('Your Report:  ')
output = ''
if user_result[0] == 0:
    output = 'You are not diabetic 🙂'
else:
    output = 'You are diabetic 🚨'

st.write(output)

#----------------- Streamlit Web App (End)----------------------------------------------------------------------------------------------------
