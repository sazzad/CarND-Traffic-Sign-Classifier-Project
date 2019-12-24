# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[train]: ./outputs/train.png "Histogram for train dataset"
[valid]: ./outputs/train.png "Histogram for validation dataset"
[test]: ./outputs/train.png "Histogram for test dataset"
[gray]: ./outputs/gray.png "Grayscaling"
[local_hist]: ./outputs/local_hist.png "Local Histogram Equalization"
[sign_0]: ./outputs/0.png "Traffic Sign 0"
[sign_17]: ./outputs/0.png "Traffic Sign 17"
[sign_24]: ./outputs/0.png "Traffic Sign 24"
[sign_28]: ./outputs/0.png "Traffic Sign 28"
[sign_40]: ./outputs/0.png "Traffic Sign 40"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. These are histogram of train, validation and test datasets target label distributions. As shows in the histogram, the datasets are not balanced. 

![alt text][train]
![alt text][valid]
![alt text][test]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because traffic signals seems to be color invariant for detection purpose.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][gray]
As a next step, I applied local histogram equalization to increase the contrast of the images.
![alt text][local_hist]

As the final step, I normalized the image data using the following equation:
  data = data - (training_data_mean) / training_data_std

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray scale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16     |
| Fully connected		| outputs 120                                   |
| RELU					|												|   
| Fully connected		| outputs 84 .                                  |
| RELU					|												|       									
| Output				| 43        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Hyperparameter   		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Learning Rate    		| 0.006   										| 
| Optimizer .        	| Adam 											|
| Epoch					| 30											|
| Batch Size			| 128											|


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.987
* validation set accuracy of 0.935
* test set accuracy of 0.907

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
	* I've chosen LeNet architecture since it achieves reasonable accuracy on the validation dataset.
* What were some problems with the initial architecture?
	* Was not achieving more than 89% of accuracy.
* How was the architecture adjusted and why was it adjusted? 
	* I've leveraged original LeNet architecture.
* Which parameters were tuned? How were they adjusted and why?
	* To improve the accuracy on validation dataset, I've tuned the following hyper parameters:
		* Learning rate, epoc size.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][sign_0] ![alt text][sign_17] ![alt text][sign_24] 
![alt text][sign_28] ![alt text][sign_40]

The first image might be difficult to classify because 2 and 7 are very close with such low resolution.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)  | Speed limit (120km/h) 							| 
| No entry     			| No entry 										|
| Road narrows on  right| General caution								|
| Children crossing	  	| Right-of-way at the next intersection			|
| Roundabout mandatory	| Roundabout mandatory      					|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is Speed limit (120km/h) (probability of 0.66), but the image  contains Speed limit (20km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .66         			| Speed limit (120km/h)   						| 
| .32     				| Speed limit (70km/h)							|
| .01					| Speed limit (20km/h)							|
| .008	      			| Wild animals crossing			 				|
| .002				    | No entry     									|


For the second image, the model is  sure that this is a No entry sign (probability 1.0), and the image does contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry   									| 
| 0     				| Speed limit (20km/h)							|
| 0					    | Speed limit (30km/h)							|
| 0	      			    | Speed limit (50km/h)			 				|
| 0				        | Speed limit (60km/h)							|

For the third image, the model is relatively sure that this is a "General caution" (probability of 0.64), but the image contains "Road narrows on the right". The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .64         			| General caution   							| 
| .36     				| Traffic signals 								|
| .0					| Pedestrians									|
| .0	      			| Road narrows on the right						|
| .0				    | Children crossing      						|

For the fourth image, the model is relatively sure that this is a "Right-of-way" sign (probability of 0.97), and the image  contain "Children crossing" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Right-of-way at the next intersection			| 
| .03     				| Children crossing 							|
|  0					| Pedestrians									|
|  0	      			| General caution				 				|
|  0				    | End of no passing by vehicles over 3.5 m.t. 	|

For the last image, the model is relatively sure that this is a "Roundabout mandatory" sign (probability of 0.91), and the image does contain "Roundabout mandatory" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			| Roundabout mandatory							| 
| .04     				| Stop 											|
| .02					| Speed limit (60km/h)							|
| .02	      			| No entry					 					|
| .01				    | Slippery road      							|

The HTML report is under: outputs/Traffic_Sign_Classifier.html

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



