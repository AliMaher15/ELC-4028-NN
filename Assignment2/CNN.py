#!/usr/bin/env python
# coding: utf-8

# In[39]:


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
from keras.layers import Dense, Flatten, InputLayer, Dropout, Conv2D, AveragePooling2D
import time
from tensorflow.keras.callbacks import EarlyStopping


# In[40]:


training_path = "Reduced_MNIST_Data\Reduced_Training_data"
testing_path = "Reduced_MNIST_Data\Reduced_Testing_data"


# In[41]:


# Define the list of classes
classes = os.listdir(training_path)


# In[42]:


print(classes)


# In[43]:


classes = list(map(int, classes))
print(classes)


# In[44]:


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


# In[45]:


# Print the shape of the data and labels arrays
print("Training Data shape:", X_train.shape)
print("Training Labels shape:", y_train.shape)


# In[46]:


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


# In[47]:


print("Testing Data shape:", X_test.shape)
print("Testing Labels shape:", y_test.shape)


# In[48]:


X_train,y_train = shuffle(X_train, y_train, random_state=4)
X_test,y_test = shuffle(X_test, y_test, random_state=4)


# In[49]:


#check if shuffling worked correctly
plt.figure()
plt.subplot(121)
plt.title("Is this {} ?".format(y_train[1050]))
plt.imshow(X_train[1050])

plt.subplot(122)
plt.title("Is this {} ?".format(y_test[1050]))
plt.imshow(X_test[1050])
plt.show()


# In[50]:


#reshaping the dataset to fit CNN architectures
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_train.shape)
print(X_test.shape)


# ## LeNet-5 - No Variations

# In[51]:


model = Sequential()

# Convolutional layer 1
model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='valid'))

# Average pooling layer 1
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Convolutional layer 2
model.add(Conv2D(16, (5, 5), activation='relu', padding='valid'))

# Average pooling layer 2
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Flatten layer
model.add(Flatten())

# Fully connected layer 1
model.add(Dense(120, activation='relu'))

# Fully connected layer 2
model.add(Dense(84, activation='relu'))

# Output layer
model.add(Dense(10, activation='softmax'))


# In[52]:


#Early Stopping to avoid fitting issues
early_stopping = EarlyStopping(monitor='accuracy', patience=3)


# In[53]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[54]:


# Train the model
tic=time.time()
model.fit(X_train, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[55]:


# Evaluate the model on the test data
tic=time.time()
test_loss, test_acc = model.evaluate(X_test, y_test)
toc=time.time()
test_time=toc-tic


# In[56]:


print("-----LeNet-5 - No Variations-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Testing Time = {} ms".format(np.round(test_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# In[ ]:





# ## Variation #1 - Adding Dropout Regularization

# In[57]:


model1 = Sequential()

# Convolutional layer 1
model1.add(Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='valid'))

#dropout regularization
model1.add(Dropout(0.2))

# Average pooling layer 1
model1.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Convolutional layer 2
model1.add(Conv2D(16, (5, 5), activation='relu', padding='valid'))

model1.add(Dropout(0.2))

# Average pooling layer 2
model1.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Flatten layer
model1.add(Flatten())

# Fully connected layer 1
model1.add(Dense(120, activation='relu'))

model1.add(Dropout(0.2))

# Fully connected layer 2
model1.add(Dense(84, activation='relu'))

model1.add(Dropout(0.2))

# Output layer
model1.add(Dense(10, activation='softmax'))


# In[58]:


model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[59]:


# Train the model
tic=time.time()
model1.fit(X_train, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[60]:


# Evaluate the model on the test data
tic=time.time()
test_loss, test_acc = model1.evaluate(X_test, y_test)
toc=time.time()
test_time=toc-tic


# In[61]:


print("-----Variation #1 - Adding Dropout-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Testing Time = {} ms".format(np.round(test_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# In[ ]:





# ## Variation #2 - Increasing Number of Filters in Conv Layers

# In[62]:


model2 = Sequential()

# Convolutional layer 1
model2.add(Conv2D(12, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='valid'))

# Average pooling layer 1
model2.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Convolutional layer 2
model2.add(Conv2D(32, (5, 5), activation='relu', padding='valid'))

# Average pooling layer 2
model2.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Flatten layer
model2.add(Flatten())

# Fully connected layer 1
model2.add(Dense(120, activation='relu'))

# Fully connected layer 2
model2.add(Dense(84, activation='relu'))

# Output layer
model2.add(Dense(10, activation='softmax'))


# In[63]:


model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[64]:


# Train the model
tic=time.time()
model2.fit(X_train, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[65]:


# Evaluate the model on the test data
tic=time.time()
test_loss, test_acc = model2.evaluate(X_test, y_test)
toc=time.time()
test_time=toc-tic


# In[66]:


print("-----Variation #2 - Increasing no. of Filters-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Testing Time= {} ms".format(np.round(test_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# In[ ]:





# ## Variation #3 - Adding "Same" Padding to Conv Layers

# In[67]:


model3 = Sequential()

# Convolutional layer 1
model3.add(Conv2D(12, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'))

# Average pooling layer 1
model3.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Convolutional layer 2
model3.add(Conv2D(32, (5, 5), activation='relu', padding='same'))

# Average pooling layer 2
model3.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Flatten layer
model3.add(Flatten())

# Fully connected layer 1
model3.add(Dense(120, activation='relu'))

# Fully connected layer 2
model3.add(Dense(84, activation='relu'))

# Output layer
model3.add(Dense(10, activation='softmax'))


# In[68]:


model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[69]:


# Train the model
tic=time.time()
model3.fit(X_train, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[70]:


# Evaluate the model on the test data
tic=time.time()
test_loss, test_acc = model3.evaluate(X_test, y_test)
toc=time.time()
test_time=toc-tic


# In[71]:


print("-----Variation #3 - Adding 'Same' Padding-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Testing Time = {} ms".format(np.round(test_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:





# In[ ]:





# ## Variation #4 - Using "Tanh" Activation

# In[72]:


model4 = Sequential()

# Convolutional layer 1
model4.add(Conv2D(12, (5, 5), activation='tanh', input_shape=(28, 28, 1), padding='valid'))

# Average pooling layer 1
model4.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Convolutional layer 2
model4.add(Conv2D(32, (5, 5), activation='tanh', padding='valid'))

# Average pooling layer 2
model4.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Flatten layer
model4.add(Flatten())

# Fully connected layer 1
model4.add(Dense(120, activation='tanh'))

# Fully connected layer 2
model4.add(Dense(84, activation='tanh'))

# Output layer
model4.add(Dense(10, activation='softmax'))


# In[73]:


model4.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[74]:


# Train the model
tic=time.time()
model4.fit(X_train, y_train, epochs=30, batch_size=32, callbacks=[early_stopping])
toc=time.time()
training_time=toc-tic


# In[75]:


# Evaluate the model on the test data
tic=time.time()
test_loss, test_acc = model4.evaluate(X_test, y_test)
toc=time.time()
test_time=toc-tic


# In[76]:


print("-----Variation #4 - Using Tanh Activation-----\n")
print("Training Time = {} s".format(np.round(training_time, 1)))
print("Testing Time = {} ms".format(np.round(test_time*1000, 1)))
print('Test Accuracy = {:.2f} %:'.format(np.round(test_acc, 3)*100))


# In[ ]:




