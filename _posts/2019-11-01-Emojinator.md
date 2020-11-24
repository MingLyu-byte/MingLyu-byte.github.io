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
	<a href="/img/emojinator/VGG16_model_summary.png"><img src="/img/emojinator/VGG16_model_summary.png"></a>
	<figcaption>Model Summary</figcaption>
</figure>

## Full Report
<object data="/assets/Projects/Emojinator_Final_Report.pdf" type="application/pdf" width="300px" height="300px">
  <embed src="/assets/Projects/Emojinator_Final_Report.pdf">
      <p>Please download the PDF to view it: <a href="/assets/Projects/Emojinator_Final_Report.pdf">Download PDF</a>.</p>
  </embed>
</object>

## Code Repo
[Link to the Code Section](https://github.com/MingLyu-byte/Emojinator/){: .btn}
