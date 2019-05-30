#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import os, os.path
import skimage



# In[ ]:


#download mnist data and split into train and test sets
def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]

    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) 
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(d)
    return images, labels

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, Y_train = load_data('data/onlinetrain')
X_test, Y_test = load_data('data/onlinetest')


# In[4]:


#plot the first image in the dataset
plt.imshow(X_train[0])


# In[5]:


#check image shape
X_train[0].shape


# In[ ]:
print (len(X_train))
print (len(Y_train))


#reshape data to fit model
X_train = X_train.reshape(len(X_train),28,28,1)
X_test = X_test.reshape(len(Y_train),28,28,1)


# In[7]:


#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[0]


# In[ ]:


#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


# In[ ]:


#compile model using accuracy as a measure of model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[10]:


#train model
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3)


# In[11]:


#show predictions for the first 3 images in the test set
model.predict(X_test[:4])


# In[12]:


#show actual results for the first 3 images in the test set
y_test[:4]

