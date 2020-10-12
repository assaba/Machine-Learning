#!/usr/bin/env python
# coding: utf-8

# # Creating Dataset
# Naive Bayes Classifier with two labels exercise

# Tutorial Link: https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

# In[1]:


weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# In[2]:


print(weather)


# # Encoding Features

# In[7]:


from sklearn import preprocessing
#creating labelEncoder
le=preprocessing.LabelEncoder()
#conver string labels into numbers
weather_encoded=le.fit_transform(weather)

print(weather_encoded)


# In[9]:


temp_encoded=le.fit_transform(temp)
play_encoded=le.fit_transform(play)
print(temp_encoded)
print(play_encoded)


# In[17]:


#combine weather and temp into a single list tuples
features=zip(weather_encoded,temp_encoded)
#Convert the zip object to a list
features = list(features)
print(features)


# In[22]:


#cearting test and training datasets
from sklearn.model_selection import train_test_split
features_train,features_test, play_train, play_test=train_test_split(features,play,test_size=0.2)


# # Model Generation

# In[23]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Create a Gaussian Classifier
clf=GaussianNB()
#fitting the model
clf.fit(features_train,play_train)
#test 
play_predict=clf.predict(features_test)

print(play_test)
print(play_predict)
#accuracy of the model
accuracy_score(play_predict,play_test)


# In[ ]:




