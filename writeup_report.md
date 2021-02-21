# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center_driving.jpg "Sample Image"
[image2]: ./writeup_images/recovery_from_left_00.jpg "Recovery Image"
[image3]: ./writeup_images/recovery_from_left_01.jpg "Recovery Image"
[image4]: ./writeup_images/recovery_from_left_02.jpg "Recovery Image"
[image5]: ./writeup_images/recovery_from_left_03.jpg "Recovery Image"
[image6]: ./writeup_images/normal_image.jpg "Normal Image"
[image7]: ./writeup_images/flipped_image.jpg "Flipped Image"
[image8]: ./writeup_images/center_driving_marked.jpg "Marked Image"
[image9]: ./writeup_images/step_01.jpg "Training Step"
[image10]: ./writeup_images/step_02.jpg "Training Step"
[image11]: ./writeup_images/step_03.jpg "Training Step"
[image12]: ./writeup_images/step_04.jpg "Training Step"

---

#### 1. Required Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Model code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it also contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. The model architecture

The model is derived from the well known paper End-to-End Deep Learning for Self-Driving Cars from Nvidia [1].

The model consists of a five convolution neural networks. 

The first three layers consists of filters of sizes 5x5 and strides (2,2) and other two layers consisting of filters of sizes of 3x3 and strides (1,1). The depths varies between 24 and 64 (model.py lines 72-76).

The model includes RELU layers to introduce nonlinearity (model.py lines 72-76) and they are added in between each of the above layers.


The model also consists of four more fully connected layers with 1164, 100, 50 and 10 neurons respectively (model.py lines 78-82).

The data is normalized in the model using a Keras lambda layer (model.py line 67). The layer normalizes the pixel data originally between 0-255 to the range -0.5 to +0.5.

In addition, the region of interest in an image is also cropped out using a Cropping Layer. ((model.py line 69)


The summary of the model is the given below. For detailed information about the model please refer [1].

| Layer (type)          |     Description / Output Shape	    		| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160,320,3 RGB image  							| 
|                       |                                               | 
| Lambda (Normalization)| 160,320,3 RGB image  							| 
| Cropping2D         	| 65,320,3 RGB image  							|
|                       |                                               | 
| Convolution 3x3     	| 1x1 stride, valid padding, Output 31x158x24 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, Output 14x77x36 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, Output 5x37x48 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, Output 3x35x64 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, Output 1x33x64 	|
| RELU					|												|
|                       |                                               |
| Flatten       		| Output 2112 									|
| Fully Connected		| Input 2112 Outputs 100						|
| Fully Connected		| Input 100 Outputs 50							|
| Fully Connected		| Input 50 Outputs 10							|
| Fully Connected		| Input 10 Outputs 1							|
| 						|												|
 
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

#### 2. Attempts to reduce overfitting in the model

The model uses the simplest mechanism of 'Early Termination/ Early Stopping' to prevent overfitting. The training is stopped once the validation accuracy saturates or starts to increase. (model.py line 62). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 94). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model uses the AdamOptimizer, so the learning rate was not tuned manually (model.py line 92).

A short overview of AdamOptimizer.

AdamOptimizer is different from **Stochastic Gradient Descent** in that SGD maintains a single learning rate and the learning rate does not change during training. [2]

AdamOptimizer uses the best of two worlds of the following two algorithms.

* **Adaptive Gradient Algorithm (AdaGrad)** maintains a per-parameter learning rate. This improves performance on problems with sparse gradients like natural language and computer vision problems. [3]

* **Root Mean Square Propagation (RMSProp)** that also maintains per-parameter learning rates. These rates are adapted based on the average of recent magnitudes of the gradients for the weight (i.e. based on rate of change). [3]


#### 4. Collecting training data

The data collection was done such that the vehicle stays mostly on the middle of the road. The image from three cameras i.e. left, center and right along with steering angle (and other data) were collected.

The first step  was to collect data when the car drives on the middle of the road. 

In the second step, data regarding the recovering from the left and right sides of the road were collected. (i.e.) The car is intentionally veered to the left and right of the road (but not recorded) and is then recovered to the middle of the road (recorded this time.)


Due to the hardware limitations, I could not collect appropriate and meaningful training data. This was evident during the training process, where reduction in validation loss does not result 

in a confident driving car. Therefore, I switched to the data provided by Udacity.


### Model Architecture and Training Strategy

#### 1. Creation of the Training Set 

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover to the center of the road when it accidently veers to the left or right side.
Without this data set, the vehicle would not have an opportunity to learn what to do when the vehicle veers off to the sides.

The below images show what a recovery looks like starting from the left side of the road to the center of the road.

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]


#### 2. Solution Design Approach

As a first step, a simple architecture that consists of a fully connected layer with neurons equal to the product of the input image parameters is created.

The aim of this step is to take the input images and create a working compilable model. 

The architecture successfully compiled and trained. The trained model is used in autonomous mode to drive the car. But the model performed poorly which is expected.


In the next step an approriate model for the given problem is selected. For this problem, I was convinced that the End-to-End Deep Learning for Self-Driving Cars from Nvidia [1] is a suitable one.

The reason is that the model addressed exactly the same use case as our project.


#### 3. Final Model Architecture

A detailed overview of the model architecture (model.py lines 18-24) is mentioned in Section 1: The model architecture.

In addition to the layers defined in the paper [1], I introduced a Cropping Layer as a preprocessing step. The aim is to train the model on only the region of the interest.

As marked in the below figure, the marked segments in the image only makes the model learn unwanted details and it may skew the training results.

Therefore, a Cropping2D Layer is introduced to the model (model.py line 69). 

![alt text][image8]


#### 4. Training Process

The model mentioned above is kept constant throughtout the training process.

In order to gauge how well the model was working, I have split my image and steering angle data into a training and validation set. 80% of the data is split as training data and the rest as validation data.


As a first step, the model is trained on a data which consists of images only from the center camera. No other data augumentation techniques is used. Only shuffling is done as a part of preprocessing.

The number of epochs is fixed at 15 since it presents ample opportunities for the model to train.

From the output graph shown below, I noticed that even though the training set loss (i.e. mean squared error) is decreasing after the sixth epoch the validation set loss started increasing,
which implied that the model was overfitting. The model was also tried on a simulator which yielded very poor performance.

![alt text][image9]

To combat the overfitting, my idea was to increase the training data. Therefore I added the images from the left and right cameras to the data set. 
The steering angle is also adjusted accordingly with a correction factor of 0.2 (model.py lines 34-43).

In this case, the training and validation loss were decreasing until the last epoch. Hence with the thought that the overfitting issue has been resolved, the model was tried on the simulator.
The model performed poorly on the right turing curves. On closer look, it is evident since we have mostly trained the network on left steered path.

![alt text][image10]

In order to avoid, the bias of the model towards left steer, the data set is augumented by adding flipped images. This introduces a right steering image for each left steering image (model.py lines 45-47) .

On training the model, it can be found that the validation loss starts increasing after eighth epoch even when traing loss was decreasing. This also implies overfitting.

![alt text][image11]

To avoid overfitting, I used the most simple technique of Early Stopping. I introduced it to the training funtion using callbacks parameter  (model.py lines 62, 94). 

This stopped by training after the ideal number of epochs which in our case is 6.

![alt text][image12]

The final model was trained on the simulator and the performance of the driving fulfilled our expectations of driving on the center of the road. 

The video file video.mp4 shows the result displayed in video format.

### Further Improvements

The project could be extended by adding the following features.
* The model fails badly on track 2. In order to overcome this, the model could be trained on a data set also generated from track 2 so that it generalises better.
* The architecture could be improved correspondingly by adding new layers.
* The layers of the neural networks could be visualised to understand more about the model. 
* Completly new architectures from reasearch and industry could also be used.

### References:

* [1] Link: [End-to-End Deep Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)
* [2] The lecture material on Stochastic Gradient Descent.
* [3] Link: [Gentle Introduction to the Adam Optimization Algorithm for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/).
* [4] The code used in this project is heavily based on the code available from exercises/assignments from the lectures.