---
title: Results and Conclusion
nav_include: 5

---

## Contents 

{:.no_toc}

[TOC]

## 1. Summary

We started using a fully connected neural network althouh we couldn't accomplish a good score for that reason we have to incorporate other techniques such as a convolitional neural network including data augmentation 

| Method                                          | Best Test Score | Best Training Score | Number of Predictors |
| ----------------------------------------------- | --------------- | ------------------- | -------------------- |
| Neural Network (Baseline)                       |                 | 3.64%               |                      |
| Convolutional Neural Network                    |                 | 18.66%              |                      |
| CNN with image augmentation                     |                 | 30.17%              |                      |
| VGG19                                           |                 | 30.15%              |                      |
| VGG19 with image augmentation                   |                 | 35.88%              |                      |
| VGG19 with image augmentation  & bounding boxes |                 | 54.17%              |                      |

## 2. Results

### 1) Fully connected network



**Fig 1. Accuracy and loss**![resnet50](/Users/jesusislas/Documents/GitHup/CS109A_Project/Images/resnet50.png)

### 2) Convolutional connected network



**Fig 2. CNN**

![Table1](/Users/jesusislas/Documents/GitHup/CS109A_Project/Images/CNN.png)

### -