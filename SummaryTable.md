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
3. [Conclusion](#summary)

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

This baseline model was able to achieve the test accuracy of 3.64 % in never seen data, using architecture of 10 fully connected layers and 100 nodes. The baseline model was run for 50 epochs saving only best weights based on cross validation accuracy.

**Fig 1. Accuracy and loss for FCN**

![Table1](/Images/dnn.png)



### 2) Convolutional connected network <a name="cnn"></a>

The next model considered was a Convolutional Neural Network, since this kind of network is well suited for image classification. We used a CNN architecture which consists of 5 ConvNets and one fully connected layer. With this approach we were able to achieve a test accuracy of 3.64 %, 



**Fig 2. Accuracy and loss for CNN**

![Table1](/Images/CNN.png)



### 3) CNN with augmentation <a name="cnn_aug"></a>

To increase the accuracy we added data augmentation to our CNN model, since we have low number of images per class. Using an image data generator, we are able to generate batches of image data with real-time data augmentation. The best accuracy that we have got after fine-tuning the CNN is around 18.66% even after having batch normalization and drop out. 



**Fig 3. Accuracy and loss for CNN with augmentation**

![Table1](/Images/CNN-aug.png)



### 4) VGG19 <a name="vgg19"></a>

Image augmentation helped CNNs to achieve better score, but to achieve higher accuracy we used a convolutional neural network ‘VGG19’ that is trained on more than a million images from the ImageNet database. The network is 19 layers deep and we have frozen all layers, except last two ConvNets and trained the model, so that we fine-tune VGG19 to our data. With the use of the VGG19 we reached a test accuracy of 30%, in never seen data. 

**Fig 4. Accuracy and loss for VGG19**

![Table1](/Images/vgg19.png)

### VGG19 with image augmentation <a name="vgg19_aug"></a>

In our next experiment, aiming to overcome the limitations of the low number of images per class, we used data augmentation combined with the inclusion of the VGG19 in the network architecture. The data augmentation helped to improve the test accuracy result to 36%, in never seen data.

**Fig 5. Accuracy and loss for VGG19 with image augmentation**

![Table1](/Images/vgg19-aug.png)



### 5) VGG19 with augmentation and cropping images using bounding boxes <a name="vgg19_box"></a>

In an attempt to further improve the accuracy results we used data augmentation and limed the images to show only the dog, cropping using bounding boxes, all these combined with the inclusion of the VGG19 in the network architecture.

The classification results obtained with our last model improved considerably, up to 54%; which we consider a very good result based on the dataset limitations and computational resources at hand. In addition the Stanford Dogs Breed dataset is known to be a difficult dataset to work with -as many people have stated in Kaggle, because of its lack of sample data per class.


**Fig 6. Accuracy and loss for VGG19 with cropping images using bouding boxes**

![Table1](/Images/vgg19-bound-aug.png)


## 3. Conclusion

Combining CNN model architecture, with the inclusion of a pre-trained model, with techniques of image data augmentation using transformations, and adding filtering using bounding boxes we were able to achieve a test data accuracy of 54%. This result is far superior to the average 27% accuracy a human expert can achieve when identifying a dog’s breed.

We feel very confident with our results; even we could not improve the accuracy any further due to computing limitations in memory and no access to use multiple GPUs. With limited computational resources that are available, we are able to include 120 dog breeds and build a model to classify them with an accuracy of 54%. For future work, with more computing resources we should be able to experiment with bigger size images and fine-tune the hyperparameters to achieve better test accuracy. 


