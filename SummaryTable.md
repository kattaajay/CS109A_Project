---
title: Results and Conclusion
nav_include: 5

---

## Contents

1. [Summary](#summary)
2. [Results](#results)
   1. [Fully Connected Network](#fcn)
   2. [Convolutional Connected Network](#cnn)
   3. [CNN with augmentation](#cnn_aug)
   4. [VGG19](#vgg19)

## 1. Summary <a name="summary"></a>

We started using a fully connected neural network, however we couldn't accomplish a good test score for that reason we incorporated other techniques such as a convolutional neural network including data augmentation.

One of the biggest challenges is to train this kind of models due to its complexity and the large image file size.  For this particular reason, we are limited to use our personal computers, jupyter portal and google colab. These last two complain about the file size so we ended up resizing which ended up paying the price with accuracy. 

![Table1](/Images/FCN.png)

Our model follows the concept depicted in the image above an image is fed into the inception model. The outcome from the inceptional model needs to go over multiple fully connected layers then the softmax returns probabilities of an image to determine which class belong to.



| Method                                          | Best Test Score | Best Training Score | Number of Predictors |
| ----------------------------------------------- | --------------- | ------------------- | -------------------- |
| Neural Network (Baseline)                       |                 | 3.64%               |                      |
| Convolutional Neural Network                    |                 | 18.66%              |                      |
| CNN with image augmentation                     |                 | 30.17%              |                      |
| VGG19                                           |                 | 30.15%              |                      |
| VGG19 with image augmentation                   |                 | 35.88%              |                      |
| VGG19 with image augmentation  & bounding boxes |                 | 54.17%              |                      |

## 2. Results <a name="results"></a>

### 1) Fully connected network <a name="fcn"></a>



**Fig 1. Accuracy and loss**

![Table1](/Images/resnet50.png)



### 2) Convolutional connected network <a name="cnn"></a>



**Fig 2. CNN**

![Table1](/Images/CNN.png)



### 3) CNN with augmentation <a name="cnn_aug"></a>



**Fig 3. CNN with augmentation**

![Table1](/Images/CNN-aug.png)



### 4) VGG19 <a name="vgg19"></a>

**Fig 4. VGG19**

![Table1](/Images/vgg19.png)