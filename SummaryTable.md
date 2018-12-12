---
title: Results and Conclusion
nav_include: 5

---

## Contents

1. [Summary](#summary)
   1. [Best model using VGG19 architecture](#Picture)
   2. [Score and paramater table per model](#Table) 
2. [Results](#results)
   1. [Fully Connected Network](#fcn)
   2. [Convolutional Connected Network](#cnn)
   3. [CNN with augmentation](#cnn_aug)
   4. [VGG19](#vgg19)
   5. [VGG19 with image augmentation](#vgg19_aug)
   6. [VGG19 with augmentation and  cropping images with bounding boxes](#vgg19_box)

## 1. Summary <a name="summary"></a>

We started using a fully connected neural network, however we couldn't accomplish a good test score for that reason we incorporated other techniques such as a convolutional neural network including data augmentation.

One of the biggest challenges is to train this kind of models due to its complexity and the large image file size.  For this particular reason and because we were using our personal computers, jupyter portal and google Colab. The model training process takes a long time, we took advantage of Colab but there is a hard limit for file size so we had to reduce the image sizes which resulted a train model with not the accuracy we were expecting but still very acceptable results. 


### 1) Best model using VGG19 architecture <a name="Picture"></a>

![Results](/Images/FCN_VGG19.png)


Our model with highest accuravy follows the concept depicted in the image above an image is fed into the VGG19 pre-trained model. The outcome from the VGG19 model needs to go over multiple fully connected layers then the softmax returns probabilities of an image to determine which class belongs to.

### 2) Score and parameter table per model <a name="Table"></a>

| Method                                          | Best Test Score | Number of Parameters |
| :---------------------------------------------- | --------------: | -------------------: |
| Neural Network (Baseline)                       |           3.64% |            2,871,920 |
| Convolutional Neural Network                    |          18.66% |            2,209,664 |
| CNN with image augmentation                     |          30.17% |            2,209,664 |
| VGG19                                           |          30.15% |           25,194,004 |
| VGG19 with image augmentation                   |          35.88% |           25,194,004 |
| VGG19 with image augmentation  & bounding boxes |          54.17% |           25,194,004 |

## 2. Results <a name="results"></a>

### 1) Fully connected network <a name="fcn"></a>

The baseline model was able to achieve the test accuracy of 3.64 % in never seen data; the next model that was considered is Convolutional Neural Networks. We used a CNN architecture which
consists of 5 ConvNets and one fully connected layer. The model was run for 100 epoch and the best weights are saved using cross-validation.

**Fig 1. Accuracy and loss for FCN**

![Table1](/Images/dnn.png)



### 2) Convolutional connected network <a name="cnn"></a>

The baseline model was able to achieve the test accuracy of 3.64 %, the next model that was considered is Convolutional Neural Networks. We CNN architecture which consists of 5 convents and one fully connected layer. The model was run for 100 epoch and the best weights are saved using cross-validation.



**Fig 2. Accuracy and loss for CNN**

![Table1](/Images/CNN.png)



### 3) CNN with augmentation <a name="cnn_aug"></a>

As we can see above, the best accuracy that we have got after fine-tuning the CNN is around 18.66% even after having batch normalization and drop out. Inorder to increase the accuracy, we did image augmentation, since we have less number of images per class. Using the image data generator that comes with keras, we are able to generate batches of image data with real-time data augmentation. The augmentation that was considered are rotation, width shift, height shift, zoom range, horizontal flip. Below are the images which have gone through multiple image augmentations.



**Fig 3. Accuracy and loss for CNN with augmentation**

![Table1](/Images/CNN-aug.png)



### 4) VGG19 <a name="vgg19"></a>

As we can see above, the best accuracy that we have got, after fine-tuning the CNN, is around 18.66% even after having batch normalization and drop out. In order to increase the accuracy, we did image augmentation, since we have low number of images per class. Using the image data generator that comes with keras, we are able to generate batches of image data with real-time data augmentation. The augmentation transformations that were considered are: rotation, width shift, height shift, zoom range, and horizontal flip. Below are the images which have gone through multiple image augmentations.

**Fig 4. Accuracy and loss for VGG19**

![Table1](/Images/vgg19.png)

### VGG19 with image augmentation <a name="vgg19_aug"></a>

With the use of the VGG19 we reached a test accuracy of 30%, in never seen data. In our next experiment, aiming to overcome the limitations of the low number of images per class.

**Fig 5. Accuracy and loss for VGG19 with image augmentation**

![Table1](/Images/vgg19-aug.png)



### 5) VGG19 with augmentation and cropping images using bounding boxes <a name="vgg19_box"></a>

The data augmentation helped to improve the test accuracy result to ~36%; it was a good improvement but still not as good as expected. In an attempt to further improve the accuracy results we used data augmentation and limed the images to show only the dog, cropping using bounding boxes, all these combined with the inclusion of the VGG19 in the network architecture.

 The classification results obtained with our last model improved considerably, up to 54%; which we consider a very good result based on the dataset limitations and computational resources at hand. In addition the Stanford Dogs Breed dataset is known to be a difficult dataset to work with -as many people have stated in Kaggle, because of its lack of sample data per class.

**Fig 6. Accuracy and loss for VGG19 with cropping images using bouding boxes**

![Table1](/Images/vgg19-bound-aug.png)

