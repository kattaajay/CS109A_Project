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
