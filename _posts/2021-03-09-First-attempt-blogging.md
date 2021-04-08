---
layout: post
comments: true
title: "First attempt blogging"
date: 2021-01-26 12:00:00
tags: machine-learning short-read
---
> In this series of posts on I will be talking about several research papers I personally like. Also, I will be writing about my hiking adventures. Stay tuned for more as I discuss about the required items, what one should be equipped with before starting hiking.

<!--more-->

### Model Workflow

How Fast R-CNN works is summarized as follows; many steps are same as in R-CNN: 
1. First, pre-train a convolutional neural network on image classification tasks.
2. Propose regions by selective search (~2k candidates per image).
3. Alter the pre-trained CNN:
	- Replace the last max pooling layer of the pre-trained CNN with a [RoI pooling](#roi-pooling) layer. The RoI pooling layer outputs fixed-length feature vectors of region proposals. Sharing the CNN computation makes a lot of sense, as many region proposals of the same images are highly overlapped.
	- Replace the last fully connected layer and the last softmax layer (K classes) with a fully connected layer and softmax over K + 1 classes.
4. Finally the model branches into two output layers:
	- A softmax estimator of K + 1 classes (same as in R-CNN, +1 is the "background" class), outputting a discrete probability distribution per RoI.
	- A bounding-box regression model which predicts offsets relative to the original RoI for each of K classes.

There are two important attributes of an image gradient:
- **Magnitude** is the L2-norm of the vector, $$g = \sqrt{ g_x^2 + g_y^2 }$$.
- **Direction** is the arctangent of the ratio between the partial derivatives on two directions, $$\theta = \arctan{(g_y / g_x)}$$.