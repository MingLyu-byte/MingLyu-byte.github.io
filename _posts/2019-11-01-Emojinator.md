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

Deep learning is a method of machine learning that uses multiple layers to automate the process of feature extraction from inputs. It has been applied in various fields, including image recognition and text analysis. For our project, we proposed a hand gesture emojinator recognitor, which takes a real-time hand image as the input and gives the predicted emoji as the output. In the training process, we applied the Convolutional neural network and VGG16. After adjusting the hyper-parameters, we get the training accuracy 1.00 and test accuracy 0.99.

## Input & Output

The input is a simple image with hand in it. The output is the predicted emoji corresponding to the hand gesture.

<figure class="third">
	<img src="/assets/img/emojinator/1_input.png">
	<img src="/assets/img/emojinator/1_intermediate.png">
	<img src="/assets/img/emojinator/1_emoji.png">
	<figcaption>Sample input, intermediate, output emoji picture</figcaption>
</figure>

We use the data from <https://github.com/akshaybahadur21/Emojinator/tree/master/gestures> and <https://github.com/akshaybahadur21/Emojinator/tree/master/hand_emo>. The idea of this project starts from the presentation in the github repo mentioned above. We want to reconstruct the application by ourselves. Thanks Akshay Bahadur and Raghav Patnecha for data source.

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

The code below display the predicted emoji. It adds a smooth transition between the emoji picture and the background so it does not look like the two are overlapping.

{% highlight python %}
{% raw %}
test_image = imagereturnshrink(emojiset,pred_class,10)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
frame_h,frame_w,frame_c = frame.shape
overlay = np.zeros((frame_h,frame_w,4),dtype='uint8')
emoji_h,emoji_w,emoji_c = test_image.shape
for i in range(0,emoji_h):
    for j in range(0,emoji_w):
        if(test_image[i,j][2] != 255 and test_image[i,j][1] != 255 and test_image[i,j][0] != 255):
            overlay[420 + i,270 + j] = test_image[i,j]
cv2.addWeighted(overlay,0.5,frame,1.0,0,frame)
{% endraw %}
{% endhighlight %}

### For further details, please see the full report or the link to the code repo.

## Full Report
<object data="/assets/Projects/Emojinator_Final_Report.pdf" type="application/pdf" width="300px" height="300px">
  <embed src="/assets/Projects/Emojinator_Final_Report.pdf">
      <p>Please download the PDF to view it: <a href="/assets/Projects/Emojinator_Final_Report.pdf">Download PDF</a>.</p>
  </embed>
</object>

## Code Repo
[Link to the Code Section](https://github.com/MingLyu-byte/Emojinator/){: .btn}
