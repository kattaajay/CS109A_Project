---
title: Introduction and EDA
notebook: Report-IntroductionandEDA.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}



## 1. Data Exploration & exploratory data analysis 

### 1) Description of Raw Data

The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. Contents of this dataset: • Number of categories: 120 • Number of images: 20,580

We will build models to classify dog breed and compare them. The files which we need to perform exploratory data analysis are

http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar

The images.tar file have all the images that are needed for our analysis and to create a model.

list.tar file have the list of all files (file_list.mat), list of training set (train_list.mat) and list of test set (test_list.mat)

Once we untar the files we would load the file_list.mat file to get the list of all files that are in the dataset.


![Experiment set-up](/Images/dog1.png)

Fig 1. Image of dog


```python

# Load the file_list.mat to get the list of all files

file_list = io.loadmat('file_list.mat')['file_list']
display(file_list)

```
