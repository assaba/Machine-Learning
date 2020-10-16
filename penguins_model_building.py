#!/usr/bin/env python
# coding: utf-8

# ####### building a classification dataset

# In[1]:


import pandas as pd
penguins = pd.read_csv('C:/Users/abder/OneDrive/Documents/Education/Python/3. Machine Learning/Data_Penguin.csv')


# In[2]:


print(penguins
    )


# In[5]:


#Ordinal feature encoding: We need to encore the qualitative information into quantitative
df=penguins.copy()
#we set the target as species as this the feature we would like to predict
target='species'
encode=['sex','island']

for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]
    
target_mapper={'Adelie':0,'Chinstrap':1,'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species']=df['species'].apply(target_encode)

#Seperating X and Y
X=df.drop('species',axis=1)
Y=df['species']


# In[7]:


#Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(X,Y)

#Saving the model
import pickle
pickle.dump(clf,open('penguins_clf.pkl','wb'))


# In[ ]:




