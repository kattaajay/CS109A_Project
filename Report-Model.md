---
title: Models
notebook: Report-Model.ipynb
nav_include: 4
---

## Contents
{:.no_toc}
*  
{: toc}

**Model Descriptions**

We used various models to for each problem formuation (3-class universal classification, 3-class subject-specific classification, 7-class universal classification, 7-class subject-specific classification, general regression, subject-specific regression). The train set consists of data from subjects 1 to 5, and the test set consists of data from subject 6. We also used two cross validation methods, normal cross validation and leave-one-subject-out cross validations. 

More information related to model training and performance can be found on the Conlusions and Results page.



## 0. Data Preparation
### 1) Reading and Cleaning Data



```python
# Load the files 
total_files = io.loadmat('file_list.mat')['file_list']
total_targets = io.loadmat('file_list.mat')['labels']


# Functionto get the paths for images
def path_to_image(img_path):
    img = image.load_img('images/'+str(img_path), target_size=(size, size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

# function to convert images to numpy arrays
def get_images(img_paths):
    list_of_images = [path_to_image(img_path[0][0]) for img_path in (img_paths)] 
    return np.vstack(list_of_images)
    

# Get all the images into numpy array and normalize
total_tensors = get_images(total_files).astype('float32')/255

# convert target label to one-hot encoding
total_targets=np.float64(total_targets)-1
total_targets = keras.utils.to_categorical(total_targets, 120)


#Split data to Train and test
X_train, X_test, y_train, y_test = train_test_split(total_tensors,
                                                    total_targets, random_state=9999,
                                                    test_size=0.2, stratify=total_targets)
```
```python
# Save the test and train data to use it later
np.save('gdrive/My Drive/Colab Notebooks/X_train', X_train)
np.save('gdrive/My Drive/Colab Notebooks/X_test', X_test)
np.save('gdrive/My Drive/Colab Notebooks/y_train', y_train)
np.save('gdrive/My Drive/Colab Notebooks/y_test', y_test)
```

## 1. Baseline Neural netwrok model

   ```python
  #Load the data

xtrain=np.load('gdrive/My Drive/Colab Notebooks/X_train.npy')
xtest=np.load('gdrive/My Drive/Colab Notebooks/X_test.npy')
ytrain=np.load('gdrive/My Drive/Colab Notebooks/y_train.npy')
ytest=np.load('gdrive/My Drive/Colab Notebooks/y_test.npy')
```
```python
# Flatten input array

X_train = xtrain.reshape(xtrain.shape[0],-1)
X_test = xtest.reshape(xtest.shape[0],-1)
```
```python
#  Run the model for 50 epochs

epochs = 50
num_hidden1 =100
num_hidden2 =100
num_hidden3 =100
num_hidden4 =100
num_hidden5 =100
num_hidden6 =100
num_hidden7 =100
num_hidden8 =100
num_hidden9 =100
num_hidden10 =100
drop_out=0.0

inputsize=X_train.shape[1]


model = Sequential()

model.add(Dense(num_hidden1, input_dim=inputsize))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(num_hidden2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(num_hidden3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(num_hidden4))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(num_hidden5))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(num_hidden6))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(num_hidden7))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(num_hidden8))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(num_hidden9))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(num_hidden10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(120, activation = 'softmax')) 

model.compile(loss='categorical_crossentropy',optimizer='RMSprop', metrics=['accuracy'])

weight_path='gdrive/My Drive/Colab Notebooks/DNN_bound.hdf5'
checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model_history = model.fit(X_train, ytrain, epochs=epochs, batch_size=128, validation_split=0.2,callbacks=callbacks_list)
```


![Models](Images/dnn.png) 

                Fig1. Loss and accuracy of Baseline model

```python
scores = model.evaluate(X_test.reshape(X_test.shape[0],-1), ytest, verbose=0)
print(" The Test accuracy for baseline model is {:2f} %".format(scores[1]*100))
```
**comment**
The Test accuracy for baseline model is is 3.644 %


## 1. Convolutional Neural Network model

