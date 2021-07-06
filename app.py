# app.py

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('hccluster.pkl', 'rb')) 
dataset= pd.read_csv('clustering dataset 2.csv')
#handling missing data
dataset.dropna(inplace=True)

def predict_voice(meanfreq, sd,	median,	IQR, skew,	kurt,	mode,	centroid,	dfrange):
  output= model.fit_predict([[0.059781, 0.064241,	0.032027,	0.075122,	12.863462,	274.402906,	0.000000,	0.059781,	0.000000,], [meanfreq, sd,	median,	IQR, skew,	kurt,	mode,	centroid,	dfrange]])
  print("Purchased", output)
  if output[1]==0:
    prediction="Female Voice"
  else:
    prediction="Male Voice"
  print(prediction)
  return prediction

def main():
    
    html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Mid Term 2</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Voice Classification using Hierarichal Clustering Algorithm")
    

    meanfreq = st.number_input('Insert Meanfreq',0,1, format="%.4f")

    sd = st.number_input('Insert SD',0,1, format="%.4f")

    median =  st.number_input('Insert median',0,1, format="%.4f")

    IQR  = st.number_input('Insert IQR',0,1, format="%.4f")

    skew   = st.number_input('Insert skew',0,100, format="%.4f")

    kurt  = st.number_input('Insert kurt',0,10, format="%.4f")

    mode = st.number_input('Insert mode',0,1, format="%.4f")

    centroid = st.number_input('Insert centroid',0,1, format="%.4f")

    dfrange = st.number_input('Insert dfrange',0,100, format="%.4f")

    result=""
    if st.button("Classify"):
      result=predict_voice(meanfreq, sd,	median,	IQR, skew,	kurt,	mode,	centroid,	dfrange)
      st.success('HC Model has predicted {}'.format(result)) 
    if st.button("About"):
      st.header("Developed by Balvinder Singh")
      st.subheader("Student , PIET")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Mid Term 2: Hc Voice Classification</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()