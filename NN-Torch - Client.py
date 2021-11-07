
# coding: utf-8

# In[64]:


#import libraries

import pandas as pd
from sklearn import preprocessing
import torch
import torch.nn as nn
import numpy as np
np.random.seed(100)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split


# In[65]:


#import dataset and print first rows
store_clients = pd.read_csv('clientp10.csv', engine='python')
store_clients.head()


# In[67]:


#replace missing values
median= store_clients['AGE'].median()
store_clients['AGE']=store_clients['AGE'].fillna(median)
sex='F'
store_clients['SESSO']=store_clients['SESSO'].fillna(sex)


# In[72]:


#encode sex attribute
store_clients['SEX']=le.fit_transform(store_clients['SESSO'])


# In[73]:


#create predictotr
predictors=['AGE','SEX','CAT','Sasi','Val','Prev']


# In[74]:


# transform dataset X and y to numpy arrays
X=store_clients[predictors].to_numpy()
y=store_clients['Label'].to_numpy()


# In[76]:


#split te dataset into test and train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, stratify=y, random_state=5)


# In[78]:


#define neural network architecture
# input layer nodes = 4 = number of features
# hidden layer nodes = 3 
# outplut layer nodes = 3 = number of categories
iln = 6
hln = 20
oln = 3
eta = 0.01
num_epoch = 60000


# In[79]:


# fc = fully connected
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(iln, hln)
        self.fc2 = nn.Linear(hln, oln)
    def forward(self, x):
        #x = torch.nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


# In[80]:


#initialize the model and the loss
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = eta)


# In[82]:


# create torch tensors for X and y
X = torch.Tensor(X_train).float()
y = torch.Tensor(y_train).long()


# In[85]:


# run the model 
for epoch in range(num_epoch): 
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    if epoch % 10000 == 0: 
        print('epoch:', epoch, 'loss:', loss.item())


# In[86]:


X = torch.Tensor(X_test).float()
y = torch.Tensor(y_test).long()


# In[88]:


# predict values
out = model(X)
(_, predicted) = torch.max(out.data, 1)


# In[92]:


#check accuracy
print('Accuracy is:', (100 * torch.sum(y == predicted).double() / len(y)))


# In[95]:


# print confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predicted )

