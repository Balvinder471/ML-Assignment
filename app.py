# app.py

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('knnmodel.pkl', 'rb')) 
dataset= pd.read_csv('Classification Dataset22.csv')
X = dataset.iloc[:, :-1].values
from sklearn.preprocessing import LabelEncoder  
label_encoder_x= LabelEncoder()  
X[:, 2]= label_encoder_x.fit_transform(X[:, 2])
X[:, 1]= label_encoder_x.fit_transform(X[:, 1])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#handling missing data
dataset.dropna(inplace=True)

def predict_exit(credit, geogrpahy, gender, age, tenure, balance, hascredit, isactive, salary):
  output = model.predict(sc.transform([[credit, geogrpahy, gender, age, tenure, balance, hascredit, isactive, salary]]))
  print("Exited", output)
  if output[1]==0:
    prediction="Customer will stay"
  else:
    prediction="Customer will exit"
  print(prediction)
  return prediction

def main():
    
    html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning End Term</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer exit prediction using KNN Algorithm")
    

    credit = st.number_input('Insert Credit Score')

    geography = st.number_input('Insert Geography France:0 Spain:1')

    gender =  st.number_input('Insert Gender Male:0 Female:1')

    age  = st.number_input('Insert Age')

    tenure   = st.number_input('Insert Tenure')

    balance  = st.number_input('Insert Balance')

    hascredit = st.number_input('Credit Card Has:1 Doesnot:0')

    isactive = st.number_input('User Active? Yes:1 Not:0')

    salary = st.number_input('Insert Salary')

    result=""
    if st.button("Classify"):
      result=predict_exit(credit, geography, gender, age, tenure, balance, hascredit, isactive, salary)
      st.success('HC Model has predicted {}'.format(result)) 
    if st.button("About"):
      st.header("Developed by Balvinder Singh")
      st.subheader("Student , PIET")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning End Term : KNN customer exit prediction</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()