---
layout: post
title:  "Heart Diease Rate Prediction"
date:   2019-11-01
excerpt: "Using Machine Learning to Predict Heart Diease Rate"
project: true
tag:
- Machine Learning
- Data Cleaning
- Python
- Jupyter Notebook
- scikit-learn
- Gradient Boost
- Random Forest
- KNN
- SVM
comments: true
---

## Introduction
In this project, we aimed to build models to predict the heart disease rate according to various features. Our dataset is from Kaggle, which collects data from counties in the United States. In all, there are 3199 data points and 33 variables in our dataset and they are from three different aspects. The first aspect which we included in our model is the area, which is categorized as rural area and urban area. The economic conditions of the counties are also considered. The economic conditions are put into six categories, including farming, mining, manufacturing, Federal/State government, recreation, and non-specialized counties. Our dataset also includes various health factors. To make the best prediction, we tried several methods, includes KNN, LinearRegression, RandomForest, Decision Tree, etc. After that, we compared the test score to determine the best model.

## Sample Data
Below is a picture of sample data input.

<figure>
	<img src="/assets/img/MLHeart/SampleData.jpg">
	<figcaption>Sample Input Data</figcaption>
</figure>

The goal is to predict the heart_disease_mortality_per_100k feature using all other column features.

## Data Preprocessing

{% highlight python %}
{% raw %}
df = df.fillna(df.mean())

features = df.drop(['heart_disease_mortality_per_100k','row_id'],axis=1)
features = pd.get_dummies(features)
target = df['heart_disease_mortality_per_100k']
features = (features - features.mean())/features.std()

df.drop(['row_id'],axis=1,inplace = True)
(train,test) = train_test_split(df,train_size=0.8,test_size=0.2)
features_train = train.drop(df.columns[0],axis=1)
features_train = pd.get_dummies(features_train)
features_train = (features_train - features_train.mean())/features_train.std()
targets_train = train.heart_disease_mortality_per_100k
features_test = test.drop(df.columns[0],axis=1)
features_test = pd.get_dummies(features_test)
features_test = (features_test - features_test.mean())/features_test.std()
targets_test = test.heart_disease_mortality_per_100k

{% endraw %}
{% endhighlight %}

## Full Report
<object data="/assets/Projects/Machine_Learning_Project_Report.pdf" type="application/pdf" width="300px" height="300px">
  <embed src="/assets/Projects/Machine_Learning_Project_Report.pdf">
      <p>Please download the PDF to view it: <a href="/assets/Projects/Machine_Learning_Project_Report.pdf">Download PDF</a>.</p>
  </embed>
</object>
