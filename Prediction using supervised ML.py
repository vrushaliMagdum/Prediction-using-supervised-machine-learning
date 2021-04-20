#!/usr/bin/env python
# coding: utf-8

# 
# # Task 1- Prediction using Supervised ML

# In[10]:


#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


#read data
data=pd.read_csv("http://bit.ly/w-data")
data.head(10)


# In[12]:


#ploting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[13]:


#Preparing the data

x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)


# In[15]:


#Training algorithm

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

print('Training Completed')


# In[17]:


#plotting the regression line

line=regressor.coef_*x+regressor.intercept_

#ploting for the test data
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# # Making Predictions

# In[18]:


print(x_test)  #Testing data- in hours
y_pred=regressor.predict(x_test) #Predicting the scores


# In[19]:


#comparing actual vs predicted

df=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
df


# In[20]:


# Test with your own data 

own_pred=regressor.predict([[9]])
print("Number of hours = {}".format(9))
print("Predicted Score = {}".format(own_pred))


# In[ ]:




