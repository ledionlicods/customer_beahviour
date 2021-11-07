#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import math
from numpy import linalg as LA
from scipy import linalg
import sklearn
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_val_score


# In[19]:


#import dataset and  print firts rows and details,missing values
dstore_clients = pd.read_csv('clusters.csv', engine='python')
dstore_clients.head()
dstore_clients.info()
dstore_clients.isnull().sum()


# In[22]:


#Fill missing values
sex='F'
dstore_clients['SESSO']=dstore_clients['SESSO'].fillna(sex)

median= dstore_clients['AGE'].median()
dstore_clients['AGE']=dstore_clients['AGE'].fillna(median)


# In[24]:


#change to 0 null values
dstore_clients['Clothes']=dstore_clients['Clothes'].fillna(0)
dstore_clients['Acc']=dstore_clients['Acc'].fillna(0)
dstore_clients['Others']=dstore_clients['Others'].fillna(0)


# In[26]:


#define Kmeans and apply it for 3 labels
def  kmeans(data, n_clusters):
    s = StandardScaler()
    data = s.fit_transform(data)
    k_m = KMeans(n_clusters=n_clusters,random_state=0)
    k_m.fit(data)
    y_pred = k_m.predict(data)
    plt.scatter(data[:,0],data[:,2],c=y_pred, cmap='Paired')
    fig = px.scatter_3d(data, x=data[:,0], y=data[:,1], z=data[:,2],labels={'x': 'Clothes', 'y': 'Accessories_Cosmetics', 'z': 'Others'},
              color=y_pred)
    print(k_m.inertia_)
    fig.show()


# In[37]:


#define Kmeans and apply it for 2 labels
def  kmeans(data, n_clusters):
    s = StandardScaler()
    data = s.fit_transform(data)
    k_m = KMeans(n_clusters=n_clusters,random_state=0)
    k_m.fit(data)
    y_pred = k_m.predict(data)
    fig = px.scatter(data, x=data[:,0], y=data[:,1],
              color=y_pred,
    labels={'x': 'Veshje', 'y': 'Aksesore'})
    fig.show()
    print(k_m.inertia_)


# In[38]:


# run K-means
kmeans(a_clients,n_clusters=4)


# In[17]:


#instal and load autorime to measute time of execution

get_ipython().system('pip install ipython-autotime')
get_ipython().run_line_magic('load_ext', 'autotime')


# In[17]:


# Run elbow method to identify best number of clusters
inertia=[]
for i in range(2,12):
    s = StandardScaler()
    data = s.fit_transform(a_clients)
    
    k_m = KMeans(n_clusters=i,random_state=0)
    k_m.fit(data)
    inertia.append(k_m.inertia_)
    
plt.plot(range(2,12), inertia)
plt.title('Metoda Elbow')
plt.xlabel('Numri i Klasterave')
plt.ylabel('Inertia')
plt.show()


# In[175]:


#Check silhouette score for the model
from sklearn.metrics import silhouette_score
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i).fit(data)
    label = kmeans.labels_
    sil_coeff = silhouette_score(data, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(i, sil_coeff))


# In[15]:


#Check correlation beetween features
klient=dstore_clients.iloc[:,4:]
correlation = klient.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Korrelimi midis Atributeve') # we plot the correlation between features.


# In[276]:


#run PCA on the features
from sklearn.decomposition import PCA
pca = PCA().fit(x_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,2,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# In[59]:


#Create a dendogram based on euclidian distance
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(a_clients, method = 'ward'))
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Klientet')
plt.ylabel('Distanca Euklidiane')
plt.show()


# In[189]:


#define agglomerative clustering
def hclustering(data, n_clusters):
    s = StandardScaler()
    data = s.fit_transform(data)
    hclustering = AgglomerativeClustering(n_clusters=n_clusters)
    y_pred = hclustering.fit_predict(data)
    #fig = px.scatter_3d(data, x=data[:,0], y=data[:,1], z=data[:,2],labels={'x': 'Clothes', 'y': 'Accessories_Cosmetics','z': 'Others'},
              #color=y_pred)
    fig = px.scatter(data, x=data[:,0], y=data[:,1],
              color=y_pred,
    labels={'x': 'Clothes', 'y': 'Accessories_Cosmetics'})    
    fig.show()


# In[ ]:


#apply agglomerative clustering
hclustering(a_clients,n_clusters=4)


# In[44]:


#define dbscan 
def dbscan(data, eps, min_samples):
    s = StandardScaler()
    data = s.fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(data)
    y_pred = db.fit_predict(data)
    #fig = px.scatter_3d(data, x=data[:,0], y=data[:,1], z=data[:,2],labels={'x': 'Clothes', 'y': 'Accessories_Cosmetics','z': 'Others'},
              #color=y_pred)
    #fig.show()
    fig = px.scatter(data, x=data[:,0], y=data[:,1], color=y_pred,
    labels={'x': 'Veshje', 'y': 'Aksesore'})    
    fig.show()
    print(len(np.unique(db.labels_)))
    for i in range(-1,len(np.unique(db.labels_))):
        print(np.where(db.labels_==i))


# In[212]:


#check number of produced clusters

s = StandardScaler()
data = s.fit_transform(a_clients)
for i in [0.1,0.2]:
    for k in [2,3,4,5,6]:
        db = DBSCAN(eps=i, min_samples=k)
        db.fit(data)
        print(len(np.unique(db.labels_)))


# In[36]:


#define and run K-medeoids
s = StandardScaler()
data = s.fit_transform(a_clients)
kmedoids = KMedoids(n_clusters=4, random_state=0).fit(data)
y_pred = kmedoids.predict(data)
plt.scatter(x=data[:,0], y=data[:,1],
              c=y_pred,cmap='Paired')
plt.title("K-Medeoids")
plt.xlabel('Veshje')
plt.ylabel('Acc')

fig = px.scatter(data, x=data[:,0], y=data[:,1],
              color=y_pred,
    labels={'x': 'Veshje', 'y': 'Aksesore'})
#fig = px.scatter_3d(data, x=data[:,0], y=data[:,1], z=data[:,2],              color=y_pred,
#labels={'x': 'Clothes', 'y': 'Accessories_Cosmetics', 'z': 'Others'})

fig.show()


# # Model Creation

# In[142]:


#import dataset and print information
store_clients = pd.read_csv('clientp7.csv', engine='python')
store_clients.info()
store_clients.head()


# In[145]:


#Check unique valyes
store_clients['SESSO'].unique()
store_clients['Label'].unique()


# In[148]:


#Histogram of Clients
tips = sns.load_dataset("tips")
ax=sns.histplot(store_clients, x="Label", stat="count", discrete=True,shrink=0.2)
ax.set(xlabel='Customer Class', ylabel='Number of Clients')


# In[151]:


#replace  null Clients age with median age
median= store_clients['AGE'].median()
store_clients['AGE']=store_clients['AGE'].fillna(median)


# In[152]:


#replace  null Clients quantity  with 0 
store_clients['Sasi']=store_clients['Sasi'].fillna(sasi)


# In[153]:


# replace missing sex value with F after pre-analysis
sex='F'
store_clients['SESSO']=store_clients['SESSO'].fillna(sex)


# In[154]:


#create predictors
predictors=['AGE','SEX','Val','Sasi','CAT','Prev']


# In[155]:


#encode sex value
le = preprocessing.LabelEncoder()
store_clients['SEX']=le.fit_transform(store_clients['SESSO'])
store_clients['SEX'].unique()


# In[158]:


# tranform data to numpy arrays
X=store_clients[predictors].to_numpy()
y=store_clients['Label'].to_numpy()


# In[172]:


#split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=5)


# In[124]:


#Apply SMOTE to deal with imbalance
import imblearn
from imblearn.over_sampling import SMOTE,SVMSMOTE
oversample = SMOTE()

X_train, y_train = oversample.fit_resample(X_train, y_train)


# In[162]:


#apply KNN model

knn=KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)


# In[164]:


#predict values
y_pred=knn.predict(X_test)


# In[166]:


#check accuracy score
accuracy_score(y_test,y_pred)


# In[168]:


# run cross validation
kfold=KFold(n_splits=5,shuffle=True)
scores=cross_val_score(knn,X_train,y_train,cv=kfold)
scores


# In[169]:


#create confusion matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn, X_test, y_test)


# In[170]:


#print metrics report
from sklearn.metrics import classification_report
report = classification_report(y_true=y_test, y_pred=y_pred)
print(report)


# In[1]:


# covert jupyter notebook to the python file 
get_ipython().system('jupyter nbconvert --to script project2.ipynb')






