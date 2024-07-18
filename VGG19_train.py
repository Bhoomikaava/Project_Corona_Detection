#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
# import tensorflow
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Input, Lambda, Dense, Flatten,Dropout
from keras.models import Model
from keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
#import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt


# In[2]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]


# In[3]:


train_path="dataset/train"
test_path="dataset/test"
val_path="dataset/val"


# In[4]:


x_train=[]

for folder in os.listdir(train_path):
    sub_path=train_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(224,224))
        x_train.append(img_arr)


# In[5]:


x_test=[]

for folder in os.listdir(test_path):
    sub_path=test_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(224,224))
        x_test.append(img_arr)
    


# In[6]:


x_val=[]

for folder in os.listdir(val_path):
    sub_path=val_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(224,224))
        x_val.append(img_arr)
    


# In[7]:


train_x=np.array(x_train)
test_x=np.array(x_test)
val_x=np.array(x_val)


# In[8]:


train_x.shape,test_x.shape,val_x.shape


# In[9]:


train_x=train_x/255.0
test_x=test_x/255.0
val_x=val_x/255.0


# In[10]:


from keras.preprocessing.image import ImageDataGenerator


# In[11]:


# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'sparse')

val_set = val_datagen.flow_from_directory(val_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'sparse')


# In[12]:


print(training_set.class_indices)


# In[13]:


train_y=training_set.classes


# In[14]:


test_y=test_set.classes


# In[15]:


val_y=val_set.classes


# In[16]:


print(train_y.shape,test_y.shape,val_y.shape)


# In[17]:


# add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[18]:


# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False


# In[19]:


# our layers - you can add more if you want
x = Flatten()(vgg.output)

prediction = Dense(3, activation='softmax')(x)


# In[20]:


# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()


# In[21]:


# tell the model what cost and optimization method to use
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)


# In[22]:


from keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)

#Early stopping to avoid overfitting of model


# In[23]:


# fit the model
history = model.fit(
  train_x,
  train_y,
  validation_data=(val_x,val_y),
  epochs=1,
  #callbacks=[early_stop],
  batch_size=16,shuffle=True)


# In[24]:


# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()

plt.savefig('vgg-loss-rps-1.png')
plt.show()


# In[25]:


# accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()

plt.savefig('vgg-acc-rps-1.png')
plt.show()


# In[26]:


model.evaluate(test_x,test_y,batch_size=32)


# In[27]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np


# In[28]:


y_pred=model.predict(test_x)
y_pred=np.argmax(y_pred,axis=1)


# In[29]:


accuracy_score(y_pred,test_y)


# In[30]:


print(classification_report(y_pred,test_y))


# In[31]:


confusion_matrix(y_pred,test_y)


# In[36]:


# path="rps-results"
# for img in os.listdir(path):
#     img=image.load_img(path+"/"+img,target_size=(224,224))
#     plt.imshow(img)
#     plt.show()
#     x=image.img_to_array(img)
#     x=np.expand_dims(x,axis=0)
#     images=np.vstack([x])
#     pred=model.predict(images,batch_size=1)
#     if pred[0][0]>0.5:
#         print("Paper")
#     elif pred[0][1]>0.5:
#         print("Rock")
#     elif pred[0][2]>0.5:
#         print("Scissors")
#     else:
#         print("Unknown")


# In[33]:


model.save("vgg-rps-final.h5")


# In[ ]:




