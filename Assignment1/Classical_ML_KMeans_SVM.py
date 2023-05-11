import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time , glob , re 
from scipy.fftpack import dct ,idct
import sklearn
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from PIL import Image


# In[2]:


path_train='D:/input/reduced-mnist/Reduced_MNIST_Data/Reduced_Trainging_data'
path_test='D:/input/reduced-mnist/Reduced_MNIST_Data/Reduced_Testing_data'

images_train=[]
images_test=[]

for i in range(10):
  # A list of all training,testing file names
  list_0 = glob.glob('{}/{}/*.jpg'.format(path_train,i)) #[:100]
  images_train.append(list_0)
  list_1 = glob.glob('{}/{}/*.jpg'.format(path_test,i)) #[:20]
  images_test.append(list_1)

# Expanding sublists into one list
images_train = [item for sublist in images_train for item in sublist]
images_test = [item for sublist in images_test for item in sublist]

#training , test data from lists
X_train = np.array([np.array(Image.open(fname)) for fname in images_train]) 
X_test = np.array([np.array(Image.open(fname)) for fname in images_test]) 

y_train=np.array([list(map(int, re.findall(r'\b\d\b', fname)))[0]  for fname in images_train]) 
y_test=np.array([list(map(int, re.findall(r'\b\d\b', fname)))[0]  for fname in images_test]) 

# Normalization for faster convergence
X_train = X_train/255
X_test = X_test/255


# DCT 

# In[3]:


# Functions used to extract DCT features

def zigzag(a):
   comp=np.concatenate([np.diagonal(a[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-a.shape[0], a.shape[0])])
   return comp[:200]


def dct_extract(a):
  features=np.zeros((a.shape[0],200))
  for i in range(a.shape[0]):
    z_features=zigzag(dct(dct(a[i].T, norm='ortho').T, norm='ortho'))
    features[i]=z_features
    extracted=features.reshape((a.shape[0],-1))
    
  return extracted


# In[4]:


X_train_DCT=dct_extract(X_train)
X_test_DCT=dct_extract(X_test)


# PCA
# 

# In[5]:


pca_model = PCA(.95) #we want a 95% variance
pca_model.fit(X_train.reshape((X_train.shape[0],28*28)))
X_train_PCA = pca_model.transform(X_train.reshape((X_train.shape[0],28*28)))
X_test_PCA = pca_model.transform(X_test.reshape((X_test.shape[0],28*28)))
print("For 95% varinace, there are {} components".format(pca_model.n_components_))


# ICA

# Independent Component Analysis is a computational method for separating a multivariate signal into additive subcomponents

# In[6]:


ica_model = FastICA(n_components=10)
X_train_ICA = ica_model.fit_transform(X_train.reshape((X_train.shape[0],784)), y_train)
X_test_ICA = ica_model.transform(X_test.reshape((X_test.shape[0],784)))


# ## K-Mean Classifiers 

# In[7]:


#calculate the accuracies
def acc_calc(y, y_hat ,c):
    y_cluster = np.zeros(y.shape)
    y_unique = np.unique(y)
    y_unique_ord = np.arange(y_unique.shape[0])
    
    for ind in range(y_unique.shape[0]):
        y[y==y_unique[ind]] = y_unique_ord[ind]

    y_unique = np.unique(y)
    bins = np.concatenate((y_unique, [np.max(y_unique)+1]), axis=0)

    for cluster in np.unique(y_hat):
        hist, _ = np.histogram(y[y_hat==cluster], bins=bins)
        correct = np.argmax(hist)
        y_cluster[y_hat==cluster] = correct
    if(c):
        return accuracy_score(y, y_cluster)
    else:
        return y_cluster


# In[8]:


#calculating the k-mean clusters
def kmean_cluster(X_train,X_test,y_test):
    no_clusters = [10,40,160,320]
    for i in no_clusters:
      kmeans = KMeans(n_clusters = i,n_init=5,max_iter=10000,algorithm='lloyd',random_state=0)
      print("Using {} clusters per class :".format(int(i/10)))
    
      start = time.time()  
      kmeans.fit(X_train)
      end=time.time()  
      print("Training Time =",end-start,"sec")  
    
      y_hat=kmeans.predict(X_test)
    
      accuracy=acc_calc(y_test, y_hat ,1)
      print("Accuracy : ",accuracy,"\n")


# DCT Results

# In[9]:


kmean_cluster(X_train_DCT,X_test_DCT,y_test)


# PCA Results

# In[10]:


kmean_cluster(X_train_PCA,X_test_PCA,y_test)


# ICA Results

# In[11]:


kmean_cluster(X_train_ICA,X_test_ICA,y_test)


# # SVM Classifiers 

# In[12]:


#using the linear kernel and the radial-basis function (rbf) non-linear kernel

def SVM_classifier(X,y,X_ts,y_ts):
  for kernel in ('linear', 'rbf'):
    model = svm.SVC(kernel=kernel, C=1)
    
    start = time.time()
    model.fit(X, y)
    end = time.time()
    
    print('Using the {} kernel: '.format(kernel))
    print("Training Time =",end-start," sec")
    
    y_hat = model.predict(X)
    y_hat_ts= model.predict(X_ts)


    print("Training Accuracy: {}".format(accuracy_score(y_hat,y)) )
    print("Testing Accuracy: {}\n".format(accuracy_score(y_hat_ts,y_ts)) )


# DCT Results

# In[13]:


SVM_classifier(X_train_DCT,y_train,X_test_DCT,y_test)


# PCA Results

# In[14]:


SVM_classifier(X_train_PCA,y_train,X_test_PCA,y_test)


# ICA Results

# In[15]:


SVM_classifier(X_train_ICA,y_train,X_test_ICA,y_test)


# Confusion Matrix For the Best Classifiers

# In[16]:


def confusion_matrix(y,y_hat):
  df = pd.DataFrame({'Labels': y, 'predictions': y_hat})
  ct = pd.crosstab(df['Labels'], df['predictions'])
  sns.heatmap(ct, annot=True, cmap='Reds', fmt='g')
  plt.xlabel('Predictions')
  plt.ylabel('Labels')
  plt.title('Confusion Matrix')
  plt.show()


# DCT-based K-means with 32 clusters

# In[17]:


kmeans_model = KMeans(n_clusters =320,n_init=5,max_iter=10000,algorithm='lloyd',random_state=0)
kmeans_model.fit(X_train_DCT)
predicted=kmeans_model.predict(X_test_DCT)
y_hat=acc_calc(y_test, predicted , 0)
confusion_matrix(y_test,y_hat)


# PCA-based SVM Classifier using the rbf kernel

# In[18]:


svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train_PCA, y_train)
predicted_svm= svm_model.predict(X_test_PCA)
confusion_matrix(y_test,predicted_svm)


# In[ ]:




