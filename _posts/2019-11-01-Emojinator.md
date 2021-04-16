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
- Jupyter Notebook
- OpenCV
- Emojis
- Tensorflow
- Keras
comments: true
---

## Introduction

Deep learning is a method of machine learning that uses multiple layers to automate the process of feature extraction from inputs. It has been applied in various fields, including image recognition and text analysis. For our project, we proposed a hand gesture emojinator recognitor, which takes a real-time hand image as the input and gives the predicted emoji as the output. In the training process, we applied the Convolutional neural network and VGG16. After adjusting the hyper-parameters, we get the training accuracy 1.00 and test accuracy 0.99.

## Input & Output

The input is a simple binary image with a hand in it. The output is the predicted emoji corresponding to the hand gesture.

<figure class="third">
	<img src="/assets/img/emojinator/1_input.png">
	<img src="/assets/img/emojinator/1_intermediate.png">
	<img src="/assets/img/emojinator/1_emoji.png">
	<figcaption>Sample input, intermediate, output emoji picture</figcaption>
</figure>

We use the data from <https://github.com/akshaybahadur21/Emojinator/tree/master/gestures> and <https://github.com/akshaybahadur21/Emojinator/tree/master/hand_emo>. The idea of this project starts from the presentation in the github repo mentioned above. We want to reconstruct the application by ourselves. Thanks Akshay Bahadur and Raghav Patnecha for data source.

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

## Load Data
We load the image data from the folder_path directory.

{% highlight python %}
{% raw %}
image_generator=ImageDataGenerator(validation_split=0.2,
                                rescale=1./255,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                vertical_flip=True)

train_generator=image_generator.flow_from_directory(folder_path,
                            target_size=(50,50),
                            color_mode='rgb',
                            batch_size=256,
                            class_mode='sparse',
                            subset="training",
                            shuffle=True)
    
validation_generator = image_generator.flow_from_directory(folder_path,
                            target_size=(50,50),
                            color_mode='rgb',
                            batch_size=256,
                            class_mode='sparse',
                            subset="validation",
                            shuffle=True)
{% endraw %}
{% endhighlight %}

## Neural Network Architecture

### Transfer Learning

We use pretrained neural network **VGG16** as basic model to develop our neural network. We choose the transfer learning method here because it saves us a lot of time from training. We also freeze the weights in the pretrained neural network by setting vgg16.layers[0].trainable = False. The summary of the model is below the code section.

{% highlight python %}
{% raw %}
vgg16 = keras.Sequential(VGG16(weights = 'imagenet',input_shape = (50,50,3),include_top = False))
vgg16.add(layers.GlobalAveragePooling2D())
vgg16.add(layers.Dense(1024,activation = 'relu'))
vgg16.add(layers.Dense(12,activation = 'softmax'))
# freeze weights of the pretrained model
vgg16.layers[0].trainable = False
vgg16.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = 'Adam',
             metrics = ['accuracy'])
vgg16.summary()
checkpt_savebest = ModelCheckpoint('Project_LeNet_V3_Epoch100.h5',save_best_only=True,verbose=2)
checkpt_earlystop = EarlyStopping(patience=2,monitor="val_loss")
hist_transfer_learning = vgg16.fit_generator(generator=train_generator,
                   validation_data=validation_generator,
                   epochs=100,
                   callbacks=[checkpt_savebest,checkpt_earlystop])
{% endraw %}
{% endhighlight %}

<figure>
	<img src="/assets/img/emojinator/VGG16_model_summary.png">
	<figcaption>Model Summary</figcaption>
</figure>

### Customized Model and Keras HyperParameter Tuning
We also tried the customized convolutional neural network and train the model from scratch with hyperparameters tuning using keras tuner. Below is the code section.

{% highlight python %}
{% raw %}
class MyHyperModel(HyperModel):

    def __init__(self, input_shape, num_classes):
        self.num_classes = num_classes
        self.input_shape = input_shape
        if(num_classes == 2):
            self.val_acc = "val_binary_accuracy"
        else:
            self.val_acc = "val_acc"
        
    def build(self, hp):
        # model build with hyperparameter tuning on all layers. Can be customized
        model = keras.Sequential()
        model.add(
                layers.Conv2D(
                filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
                kernel_size=3,
                activation='relu',
                input_shape=self.input_shape
            )
        )
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(
            layers.Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
                )
            )
        )
        model.add(
                layers.Conv2D(
                filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
                kernel_size=3,
                activation='relu'
            )
        )
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(
            layers.Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
                )
            )
        )
        model.add(
                layers.Conv2D(
                filters=hp.Int('conv_2_filter', min_value=16, max_value=32, step=16),
                kernel_size=3,
                activation='relu'
            )
        )
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(
            layers.Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
                )
            )
        )
        model.add(
                layers.Conv2D(
                filters=hp.Int('conv_2_filter', min_value=16, max_value=32, step=16),
                kernel_size=3,
                activation='relu'
            )
        )
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(
            layers.Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
                )
            )
        )
        model.add(layers.Flatten())
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=64,
                                            max_value=512,
                                            step=32)))
                  
        # binary classification or multiclass classificatio based on number of classes
        if(self.num_classes == 2):
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4,1e-5])),
                    loss='binary_crossentropy',
                    metrics=[keras.metrics.binary_accuracy])
            
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            model.compile(optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4,1e-5])),
                    loss='sparse_categorical_crossentropy',
                    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
        return model
{% endraw %}
{% endhighlight %}

This is the custimized model class using keras tuning. We can set search space for values such as filter numbers, learning rate and etc. Keras tuner helps automating the process
of hyperparameter tuning. 

{% highlight python %}
{% raw %}
seed = 1
exe_per_trial = 2
max_epochs = 10
n_epoch_search = 50
output_dir = 'hyperparameter'
project_name = 'Emojinator'
hypermodel = MyHyperModel(num_classes = 12,input_shape = (50,50,3))
tuner = Hyperband(
        hypermodel,
        objective = hypermodel.val_acc,
        seed = 10,
        executions_per_trial = exe_per_trial,
        directory = output_dir,
        project_name = project_name,
        max_epochs = max_epochs
)

train_img,train_lables = train_generator.next()
test_img,test_lables = validation_generator.next()
tuner.search(train_img,train_lables, 
             epochs = n_epoch_search,
             validation_data=(test_img,test_lables),
             batch_size = 256)
             
# Show a summary of the search
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save(str(project_name + ".h5"))
    
model = keras.models.load_model(str(project_name + ".h5"))
score = model.evaluate(test_img,test_lables, verbose=0)
print(model.metrics_names)
print(score)
print(score)
{% endraw %}
{% endhighlight %}

There are also several hyperparameters that we can set to the tuner. Since we have limited computing resources, we set a limit to the number of combinations that we are going to try and also to the number of epochs we get to train on each combination. The keras tuner will give us the best model based on the evaluation metrics selected.

## OpenCV and Display of Emoji

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

## Live Demo

This is a short live demo of our project. The prediction is not stable as it is in the train and testing stage since we are feeding directly from video. Overall, we can see that most of the time the model outputs the correct prediction.

![Alt Text](/assets/Projects/Emojinator_Live_Demo.gif)

<!-- ### For further details, please see the full report or the link to the code repo. -->

<!-- ## Full Report
<object data="/assets/Projects/Emojinator_Final_Report.pdf" type="application/pdf" width="300px" height="300px">
  <embed src="/assets/Projects/Emojinator_Final_Report.pdf">
      <p>Please download the PDF to view it: <a href="/assets/Projects/Emojinator_Final_Report.pdf">Download PDF</a>.</p>
  </embed>
</object> -->

## Code Repo
[Link to the Code Section](https://github.com/MingLyu-byte/Emojinator/){: .btn}
