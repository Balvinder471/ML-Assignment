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
    

    meanfreq = st.text_input('Insert Meanfreq')

    sd = st.text_input('Insert SD')

    median =  st.text_input('Insert median')

    IQR  = st.text_input('Insert IQR')

    skew   = st.text_input('Insert skew')

    kurt  = st.text_input('Insert kurt')

    mode = st.text_input('Insert mode')

    centroid = st.text_input('Insert centroid')

    dfrange = st.text_input('Insert dfrange')

    result=""
    if st.button("Classify"):
      result=predict_voice(float(meanfreq), float(sd), float(median),	float(IQR), float(skew),float(kurt),float(mode),float(centroid),float(dfrange))
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