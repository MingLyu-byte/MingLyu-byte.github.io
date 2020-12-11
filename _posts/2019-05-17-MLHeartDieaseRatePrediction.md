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

The goal is to predict the heart_disease_mortality_per_100k feature using all other features.

## Data Preprocessing

{% highlight python %}
{% raw %}
df = df.fillna(df.mean())

features = df.drop(['heart_disease_mortality_per_100k','row_id'],axis=1)
features = pd.get_dummies(features)
target = df['heart_disease_mortality_per_100k']

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

## Lasso Regression

{% highlight python %}
{% raw %}
grid = {'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]}
lasso = Lasso()
lassoCV = GridSearchCV(lasso,param_grid=grid,return_train_score=True)
lassoCV.fit(features_train,targets_train)

print()
print("best alpha",lassoCV.best_params_,'test_R2',lassoCV.best_score_)
performance = pd.DataFrame()
performance['alpha'] = np.log10(grid['alpha'])
performance['train_R2'] = lassoCV.cv_results_['mean_train_score'] 
performance['test_R2'] = lassoCV.cv_results_['mean_test_score'] 

ax1 = performance.plot.line(x = 'alpha',y='train_R2')
ax = performance.plot.line(x = 'alpha',y='test_R2',ax = ax1)

la = lassoCV.best_estimator_
coef = pd.Series(la.coef_,index = features_train.columns)
coef.nonzero().sort_values()
{% endraw %}
{% endhighlight %}

## Gradient Boost

{% highlight python %}
{% raw %}
time_start = time()
grid = {'n_estimators':np.arange(10,100,10),'max_depth':np.arange(1,10,1),'learning_rate':np.arange(0.01,1,0.1)}

gb = GradientBoostingRegressor()
gbCV = GridSearchCV(gb,param_grid=grid,return_train_score=True,n_jobs=-1)
gbCV.fit(features_train,targets_train)
time_stop = time()
time_elapsed = (time_stop - time_start)/60.0
print('time_elapsed =',round(time_elapsed,1),'min')

print()
print(gbCV.best_params_,',validation R2 =',gbCV.best_score_.round(3))

gb = gbCV.best_estimator_
R2_train = gb.score(features_train,targets_train)
R2_test  = gb.score(features_test,targets_test)
print('train R2 =',R2_train.round(3),'test R2 =',R2_test.round(3))
{% endraw %}
{% endhighlight %}

We fullfill all missing values with the mean value. The reason is given in the report. Then, we do a train and test split with respective ratios of 0.8 and 0.2. We also do a normalization on the feature values. Now, we can feed the data into machine learning models.

## Full Report
<object data="/assets/Projects/Machine_Learning_Project_Report.pdf" type="application/pdf" width="300px" height="300px">
  <embed src="/assets/Projects/Machine_Learning_Project_Report.pdf">
      <p>Please download the PDF to view it: <a href="/assets/Projects/Machine_Learning_Project_Report.pdf">Download PDF</a>.</p>
  </embed>
</object>
