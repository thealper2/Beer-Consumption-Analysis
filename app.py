import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("linreg.pkl", "rb"))

st.title("Beer Consumption Analysis")
medi = st.number_input("Medium Temperature")
mini = st.number_input("Minimum Temperature")
maxi = st.number_input("Maximum Temperature")
prec = st.number_input("Precipitacao")
endw = st.number_input("End of Week")

if st.button("Predict"):
    test = np.array([[medi, mini, maxi, prec, endw]])
    res = model.predict(test)
    print(res)
    st.success("Predicted: " + str(res[0]))
