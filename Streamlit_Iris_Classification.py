#!/usr/bin/env python
# coding: utf-8

# # Web application with Streamlit and Iris Classification

# In[ ]:


#How to run streamlit after saving the file: streamlit run [filename].py


# In[9]:


#Imports
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB


# In[ ]:


st.write("""
#Iris flower prediction app
This app predicts the Iris flower type
""")


# In[2]:


#Creation of a sidebar to allow user input parameters
st.sidebar.header('User Input Parameters')

#function to allow the user to input the featurs
def user_input_features():
    sepal_length = st.slider('sepal_length',4.3,7.9,5.4)
    sepal_width = st.slider('sepal_length',2.0,4.4,3.4)
    sepal_length = st.slider('sepal_length',1.0,6.9,1.3)
    sepal_width = st.slider('sepal_length',0.1,2.5,0.2)
    data={'sepal_length':sepal_length,
          'sepal_length':sepal_width,
          'sepal_length':sepal_length,
          'sepal_length':sepal_width,  
    }
    features=pd.DataFrame(data,index=[0])
    return features


# In[5]:


#Outputs in the web app
df=user_input_features()

st.subheader('User Input parameters')
st.write(df)


# In[10]:


#Model
iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = GaussianNB()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


# In[11]:


#Description of the results in the web app
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:




