# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
note that I've modified the drive.py file to drive at 30mph instead of 9mph (because I'm impatient)

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48 (model.py lines 102-106) and 2 3x3 filter sizes of 64 depth.  All use RELU activation.

The model includes Fully connected RELU layers with L2 regularization to introduce nonlinearity (model.py lines 111-115), and the data is normalized in the model using a Keras lambda layer (code line 99). 

#### 2. Attempts to reduce overfitting in the model

I tried using dropout, but the model performed worse (maybe I should have used more epochs?)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, learning rate set to 0.0001

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also rode the track in the opposite direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used the NVIDIA network architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to use Dropout. the validation set had a lower error rate, but overall the error rate went up. My test drive had my car driving off road almost intentionally.  I decided to give it more training data.  It still kept jumping off the road at a few places. So, I decided to use the left and right cameras. It actually performed worse.  I tried pooling also, but this didn't seem to help.

I spent about 20 minutes collecting training data of vehicle recovering from the side of the road. This helped in a few cases, but it still wanted to drive offroad whenever the barriers were missing. I finally tried removing the dropout and then my mse went down and the vehicle stayed on the road.

#### 2. Final Model Architecture

Lesson learned: this exercise relies **heavily** on training data. Variations, flipping the images (randomly works better than every time), and lots of recovery examples.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. 

I also recorded two laps driving the course backwards.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to 

I repeated this process on track two in order to get more data points, but my model performed worse.  so, I removed that data.

To augment the data sat, I also flipped images and angles randomly during the generation process.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20.

Here's a histogram of the training progress over 20 epochs:

![alt text][image1]
