---
layout: post
title:  "Deep Learning Emojinator"
date:   2019-11-01
excerpt: "Transform Hand Gestures into Emojis Using Deep Learning"
project: true
tag:
- Deep Learning
- Image Processing
- Python
- Juypter Notebook
- OpenCV
- Emojis
comments: true
---

## Introduction

## Input & Output

The input is a simple image with hand in it. The output is the predicted emoji corresponding to the hand gesture.

<figure class="third">
	<img src="/assets/img/emojinator/1_input.png">
	<img src="/assets/img/emojinator/1_intermediate.png">
	<img src="/assets/img/emojinator/1_emoji.png">
	<figcaption>Sample input, intermediate, output emoji picture</figcaption>
</figure>


## Neural Network Architecture

We use pretrained neural network **VGG16** as basic model to develop our neural network. We choose the transfer learning method here because it saves us a lot of time from training. We also freeze the weights in the pretrained neural network by setting vgg16.layers[0].trainable = False. The summary of the model is below the code section.

{% highlight python %}
{% raw %}
vgg16 = Sequential(VGG16(weights = 'imagenet',input_shape = (50,50,3),include_top = False))
vgg16.add(Flatten())
vgg16.add(Dense(12,activation = 'relu'))
vgg16.add(Dense(12,activation = 'softmax'))
vgg16.layers[0].trainable = False
vgg16.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = 'Adam',
             metrics = ['accuracy'])
vgg16.summary()
{% endraw %}
{% endhighlight %}

<figure>
	<img src="/assets/img/emojinator/VGG16_model_summary.PNG">
	<figcaption>Model Summary</figcaption>
</figure>

## Image Processing

We set some color thresholds to that differ the hand from the background. There are couple image processing methods we use here including erosion, dilation and gaussian filters. The purpose of this step is to extract the hand part as much as possible and also excludes unnecessary parts.

{% highlight python %}
{% raw %}
hsv = cv2.cvtColor(image_capture, cv2.COLOR_BGR2HSV)
lower_skin = np.array([0,15,70], dtype=np.uint8)
upper_skin = np.array([20,150,255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_skin, upper_skin)
mask = cv2.erode(mask,kernel,iterations = 2)
mask = cv2.dilate(mask,kernel,iterations = 7)
mask = cv2.GaussianBlur(mask,(3,3),100)
{% endraw %}
{% endhighlight %}

The examples are presented in the report.



## OpenCV and Live Demo

## Full Report
<object data="/assets/Projects/Emojinator_Final_Report.pdf" type="application/pdf" width="300px" height="300px">
  <embed src="/assets/Projects/Emojinator_Final_Report.pdf">
      <p>Please download the PDF to view it: <a href="/assets/Projects/Emojinator_Final_Report.pdf">Download PDF</a>.</p>
  </embed>
</object>

## Code Repo
[Link to the Code Section](https://github.com/MingLyu-byte/Emojinator/){: .btn}
