#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


df = pd.read_csv('dhaka price.csv')


# In[5]:


df


# In[6]:


df.head(3)


# In[7]:


x = df[['area']]
y = df['price']


# In[8]:


x


# In[9]:


y


# In[17]:


plt.scatter(df['area'],df['price'], marker="+", color="red")
plt.xlabel('Area in sq feet')
plt.ylabel('price in taka')
plt.title('Home prices in Dhaka')


# In[22]:


from sklearn.model_selection import train_test_split


# In[31]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=1)


# In[30]:


xtrain


# In[28]:


ytrain


# In[32]:


xtest


# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


reg = LinearRegression()


# In[35]:


reg.fit(xtrain,ytrain)


# In[36]:


reg.predict(xtest)


# In[39]:


plt.scatter(df['area'],df['price'], marker="+", color="red")
plt.xlabel('Area in sq feet')
plt.ylabel('price in taka')
plt.title('Home prices in Dhaka')
plt.plot(df.area, reg.predict(df[['area']]))


# In[44]:


reg.predict([[37000]])


# In[42]:


reg.coef_


# In[43]:


reg.intercept_


# In[45]:


y = 15.16134343*37000 + 5990.615739216344


# In[46]:


y


# In[ ]:




