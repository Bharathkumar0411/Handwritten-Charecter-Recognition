#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dfx=pd.read_csv('xdata.csv')
dfy=pd.read_csv('ydata.csv')


# In[3]:


X=dfx.values
Y=dfy.values


# In[5]:


print(X.shape)
Y.shape


# In[6]:


X=X[:,1:]
Y=Y[:,1:].reshape((-1,))


# In[7]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[8]:


query_x=np.array([2,3])
plt.scatter(X[:,0],X[:,1],c=Y)
plt.scatter(query_x[0],query_x[1],color='red')
plt.show()


# In[9]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))


# In[15]:


def knn(X,Y,querypoint,k=5):
    vals=[]
    m=X.shape[0]
    for i in range(m):
        d=dist(querypoint,X[i])
        vals.append((d,Y[i]))
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals)
    #to show occurance of each number
    new_vals=np.unique(vals[:,1],return_counts=True)
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    return pred


# In[16]:


knn(X,Y,query_x)


# MNIST Dataset

# In[20]:


df=pd.read_csv('train.csv')
df.head()


# In[21]:


df.shape


# In[22]:


df.columns


# In[23]:


data=df.values
data.shape


# In[24]:


type(data)


# In[25]:


X=data[:,1:]
Y=data[:,0]


# In[26]:


X.shape


# In[27]:


Y.shape


# In[28]:


split=int(0.9*X.shape[0])
print(split)


# In[29]:


X_train=X[:split,:]
Y_train=Y[:split]
X_test=X[split:,:]
Y_test=Y[split:]


# In[30]:


def drawImg(sample):
    img=sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()


# In[33]:


n=np.random.randint(1,500)
drawImg(X_train[n])
print(Y_train[n])


# In[36]:


n=np.random.randint(1,100)
pred=knn(X_train,Y_train,X_test[n])
print(pred)


# In[41]:


n=np.random.randint(1,100)
drawImg(X_test[0])
print(Y_test[0])


# In[ ]:




