#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from scipy.fftpack import dct ,idct
import sklearn
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer, Dropout
import time
from tensorflow.keras.callbacks import EarlyStopping


# In[19]:


training_path = "Reduced_MNIST_Data\Reduced_Training_data"
testing_path = "Reduced_MNIST_Data\Reduced_Testing_data"


# In[20]:


# Define the list of classes
classes = os.listdir(training_path)


# In[21]:


print(classes)


# In[22]:


classes = list(map(int, classes))
print(classes)


# In[23]:


# Define an empty list to store the data and labels
X_train = []
y_train = []

# Loop over the classes
for class_name in classes:
    class_path = os.path.join(training_path, str(class_name))
    # Loop over the images in the class folder
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        # Load the image and append it to the data list
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        X_train.append(image)
        # Append the label to the labels list
        y_train.append(class_name)

# Convert the data and labels lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)


# In[24]:


# Print the shape of the data and labels arrays
print("Training Data shape:", X_train.shape)
print("Training Labels shape:", y_train.shape)


# In[25]:


X_test = []
y_test = []

for class_name in classes:
    class_path = os.path.join(testing_path, str(class_name))

    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        X_test.append(image)
        y_test.append(class_name)

# Convert the data and labels lists to NumPy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)


# In[26]:


print("Testing Data shape:", X_test.shape)
print("Testing Labels shape:", y_test.shape)


# In[27]:


X_train,y_train = shuffle(X_train, y_train, random_state=4)
X_test,y_test = shuffle(X_test, y_test, random_state=4)


# In[28]:


#check if shuffling worked correctly
plt.figure()
plt.subplot(121)
plt.title("Is this {} ?".format(y_train[1050]))
plt.imshow(X_train[1050])

plt.subplot(122)
plt.title("Is this {} ?".format(y_test[1050]))
plt.imshow(X_test[1050])
plt.show()


# ## DCT Features

# In[29]:


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


# In[30]:


#Extract DCT features for training and testing data
X_train_DCT=dct_extract(X_train)
X_test_DCT=dct_extract(X_test)


# In[31]:


X_train_DCT.shape


# ## PCA Features

# In[32]:


pca_model = PCA(.95) #we want a 95% variance
pca_model.fit(X_train.reshape((X_train.shape[0],28*28)))
X_train_PCA = pca_model.transform(X_train.reshape((X_train.shape[0],28*28)))
X_test_PCA = pca_model.transform(X_test.reshape((X_test.shape[0],28*28)))
print("For 95% varinace, there are {} components".format(pca_model.n_components_))


# In[33]:


X_train_PCA.shape


# ## ICA Features

# In[34]:


ica_model = FastICA(n_components=200)
X_train_ICA = ica_model.fit_transform(X_train.reshape((X_train.shape[0],784)), y_train)
X_test_ICA = ica_model.transform(X_test.reshape((X_test.shape[0],784)))


# In[35]:


X_train_ICA.shape


# # Training a Multi-layer Perceptron (MLP)

# ##     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Using DCT Features

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 Hidden Layer

# In[36]:


# Define the model architecture
model_MLP1_DCT = Sequential(name='MLP1_DCT')

model_MLP1_DCT.add(Dense(256, activation='relu', input_shape=(200,)))  # hidden layer
model_MLP1_DCT.add(Dropout(0.2)) #dropout regularization
model_MLP1_DCT.add(Dense(10, activation='softmax'))  # Output layer

model_MLP1_DCT.summary()


# In[37]:


#Early Stopping to avoid fitting issues
early_stopping = EarlyStopping(monitor='accuracy', patience=3)


# In[38]:


# Compile the model
model_MLP1_DCT.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[39]:


# Train the model
tic=time.time()
model_MLP1_DCT.fit(X_train_DCT, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[40]:


# Evaluate the model on the test data
test_loss, test_acc = model_MLP1_DCT.evaluate(X_test_DCT, y_test)


# In[41]:


X_test_DCT[0].shape


# In[42]:


tic=time.time()
model_MLP1_DCT.predict(X_test_DCT[0].reshape(1,200))
toc=time.time()
proc_time=toc-tic


# In[43]:


print("-----MLP With 1 Hidden Layer-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Processing Time for 1 example = {} ms".format(np.round(proc_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2 Hidden Layers

# In[44]:


# Define the model architecture
model_MLP2_DCT = Sequential(name='MLP2_DCT')

model_MLP2_DCT.add(Dense(256, activation='relu', input_shape=(200,)))  # 1st hidden layer
model_MLP2_DCT.add(Dropout(0.2))
model_MLP2_DCT.add(Dense(128, activation='relu'))  # 2nd hidden layer
model_MLP2_DCT.add(Dropout(0.2))
model_MLP2_DCT.add(Dense(10, activation='softmax'))  # Output layer

model_MLP2_DCT.summary()


# In[45]:


# Compile the model
model_MLP2_DCT.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[46]:


# Train the model
tic=time.time()
model_MLP2_DCT.fit(X_train_DCT, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[47]:


# Evaluate the model on the test data
test_loss, test_acc = model_MLP2_DCT.evaluate(X_test_DCT, y_test)


# In[48]:


tic=time.time()
model_MLP2_DCT.predict(X_test_DCT[0].reshape(1,200))
toc=time.time()
proc_time=toc-tic


# In[49]:


print("-----MLP With 2 Hidden Layers-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Processing Time for 1 example = {} ms".format(np.round(proc_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3 Hidden Layers

# In[50]:


# Define the model architecture
model_MLP3_DCT = Sequential(name='MLP3_DCT')

model_MLP3_DCT.add(Dense(256, activation='relu', input_shape=(200,)))  # 1st hidden layer
model_MLP3_DCT.add(Dropout(0.2))
model_MLP3_DCT.add(Dense(128, activation='relu'))  # 2nd hidden layer
model_MLP3_DCT.add(Dropout(0.2))
model_MLP3_DCT.add(Dense(64, activation='relu'))  # 3rd hidden layer
model_MLP3_DCT.add(Dropout(0.2))
model_MLP3_DCT.add(Dense(10, activation='softmax'))  # Output layer

model_MLP3_DCT.summary()


# In[51]:


# Compile the model
model_MLP3_DCT.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[52]:


# Train the model
tic=time.time()
model_MLP3_DCT.fit(X_train_DCT, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[53]:


# Evaluate the model on the test data
test_loss, test_acc = model_MLP3_DCT.evaluate(X_test_DCT, y_test)


# In[54]:


tic=time.time()
model_MLP3_DCT.predict(X_test_DCT[0].reshape(1,200))
toc=time.time()
proc_time=toc-tic


# In[55]:


print("-----MLP With 3 Hidden Layers-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Processing Time for 1 example = {} ms".format(np.round(proc_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# In[ ]:





# In[ ]:





# ##     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Using PCA Features

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 Hidden Layer

# In[56]:


# Define the model architecture
model_MLP1_PCA = Sequential(name='MLP1_PCA')

model_MLP1_PCA.add(Dense(256, activation='relu', input_shape=(262,)))  # hidden layer
model_MLP1_PCA.add(Dropout(0.2)) #dropout regularization
model_MLP1_PCA.add(Dense(10, activation='softmax'))  # Output layer

model_MLP1_PCA.summary()


# In[57]:


# Compile the model
model_MLP1_PCA.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[58]:


# Train the model
tic=time.time()
model_MLP1_PCA.fit(X_train_PCA, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[59]:


# Evaluate the model on the test data
test_loss, test_acc = model_MLP1_PCA.evaluate(X_test_PCA, y_test)


# In[60]:


tic=time.time()
model_MLP1_PCA.predict(X_test_PCA[0].reshape(1,262))
toc=time.time()
proc_time=toc-tic


# In[61]:


print("-----MLP With 1 Hidden Layer-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Processing Time for 1 example = {} ms".format(np.round(proc_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2 Hidden Layers

# In[62]:


# Define the model architecture
model_MLP2_PCA = Sequential(name='MLP2_PCA')

model_MLP2_PCA.add(Dense(256, activation='relu', input_shape=(262,)))  # 1st hidden layer
model_MLP2_PCA.add(Dropout(0.2))
model_MLP2_PCA.add(Dense(128, activation='relu'))  # 2nd hidden layer
model_MLP2_PCA.add(Dropout(0.2))
model_MLP2_PCA.add(Dense(10, activation='softmax'))  # Output layer

model_MLP2_PCA.summary()


# In[63]:


# Compile the model
model_MLP2_PCA.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[64]:


# Train the model
tic=time.time()
model_MLP2_PCA.fit(X_train_PCA, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[65]:


# Evaluate the model on the test data
test_loss, test_acc = model_MLP2_PCA.evaluate(X_test_PCA, y_test)


# In[66]:


tic=time.time()
model_MLP2_PCA.predict(X_test_PCA[0].reshape(1,262))
toc=time.time()
proc_time=toc-tic


# In[67]:


print("-----MLP With 2 Hidden Layers-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Processing Time for 1 example = {} ms".format(np.round(proc_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3 Hidden Layers

# In[68]:


# Define the model architecture
model_MLP3_PCA = Sequential(name='MLP3_PCA')

model_MLP3_PCA.add(Dense(256, activation='relu', input_shape=(262,)))  # 1st hidden layer
model_MLP3_PCA.add(Dropout(0.2))
model_MLP3_PCA.add(Dense(128, activation='relu'))  # 2nd hidden layer
model_MLP3_PCA.add(Dropout(0.2))
model_MLP3_PCA.add(Dense(64, activation='relu'))  # 3rd hidden layer
model_MLP3_PCA.add(Dropout(0.2))
model_MLP3_PCA.add(Dense(10, activation='softmax'))  # Output layer

model_MLP3_PCA.summary()


# In[69]:


# Compile the model
model_MLP3_PCA.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[70]:


# Train the model
tic=time.time()
model_MLP3_PCA.fit(X_train_PCA, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[71]:


# Evaluate the model on the test data
test_loss, test_acc = model_MLP3_PCA.evaluate(X_test_PCA, y_test)


# In[72]:


tic=time.time()
model_MLP3_PCA.predict(X_test_PCA[0].reshape(1,262))
toc=time.time()
proc_time=toc-tic


# In[73]:


print("-----MLP With 3 Hidden Layers-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Processing Time for 1 example = {} ms".format(np.round(proc_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# In[ ]:





# In[ ]:





# ##     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Using ICA Features

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 Hidden Layer

# In[74]:


# Define the model architecture
model_MLP1_ICA = Sequential(name='MLP1_ICA')

model_MLP1_ICA.add(Dense(256, activation='relu', input_shape=(200,)))  # hidden layer
model_MLP1_ICA.add(Dropout(0.2)) #dropout regularization
model_MLP1_ICA.add(Dense(10, activation='softmax'))  # Output layer

model_MLP1_ICA.summary()


# In[75]:


# Compile the model
model_MLP1_ICA.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[76]:


# Train the model
tic=time.time()
model_MLP1_ICA.fit(X_train_ICA, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[77]:


# Evaluate the model on the test data
test_loss, test_acc = model_MLP1_ICA.evaluate(X_test_ICA, y_test)


# In[78]:


tic=time.time()
model_MLP1_ICA.predict(X_test_ICA[0].reshape(1,200))
toc=time.time()
proc_time=toc-tic


# In[79]:


print("-----MLP With 1 Hidden Layer-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Processing Time for 1 example = {} ms".format(np.round(proc_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2 Hidden Layers

# In[80]:


# Define the model architecture
model_MLP2_ICA = Sequential(name='MLP2_ICA')

model_MLP2_ICA.add(Dense(256, activation='relu', input_shape=(200,)))  # 1st hidden layer
model_MLP2_ICA.add(Dropout(0.2))
model_MLP2_ICA.add(Dense(128, activation='relu'))  # 2nd hidden layer
model_MLP2_ICA.add(Dropout(0.2))
model_MLP2_ICA.add(Dense(10, activation='softmax'))  # Output layer

model_MLP2_ICA.summary()


# In[81]:


# Compile the model
model_MLP2_ICA.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[82]:


# Train the model
tic=time.time()
model_MLP2_ICA.fit(X_train_ICA, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[83]:


# Evaluate the model on the test data
test_loss, test_acc = model_MLP2_ICA.evaluate(X_test_ICA, y_test)


# In[84]:


tic=time.time()
model_MLP2_ICA.predict(X_test_ICA[0].reshape(1,200))
toc=time.time()
proc_time=toc-tic


# In[85]:


print("-----MLP With 2 Hidden Layers-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Processing Time for 1 example = {} ms".format(np.round(proc_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3 Hidden Layers

# In[86]:


# Define the model architecture
model_MLP3_ICA = Sequential(name='MLP3_ICA')

model_MLP3_ICA.add(Dense(256, activation='relu', input_shape=(200,)))  # 1st hidden layer
model_MLP3_ICA.add(Dropout(0.2))
model_MLP3_ICA.add(Dense(128, activation='relu'))  # 2nd hidden layer
model_MLP3_ICA.add(Dropout(0.2))
model_MLP3_ICA.add(Dense(64, activation='relu'))  # 3rd hidden layer
model_MLP3_ICA.add(Dropout(0.2))
model_MLP3_ICA.add(Dense(10, activation='softmax'))  # Output layer

model_MLP3_ICA.summary()


# In[87]:


# Compile the model
model_MLP3_ICA.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[88]:


# Train the model
tic=time.time()
model_MLP3_ICA.fit(X_train_ICA, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[89]:


# Evaluate the model on the test data
test_loss, test_acc = model_MLP3_ICA.evaluate(X_test_ICA, y_test)


# In[90]:


tic=time.time()
model_MLP3_ICA.predict(X_test_ICA[0].reshape(1,200))
toc=time.time()
proc_time=toc-tic


# In[91]:


print("-----MLP With 3 Hidden Layers-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Processing Time for 1 example = {} ms".format(np.round(proc_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:




