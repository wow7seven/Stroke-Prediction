**Stroke Prediction**

Hriday Kondru (B20CS021), Rachit (B20AI032), Rohit Bhanudas Kote(B20CS056)

**Abstract:**

This paper reports our experience with building a Stroke prediction machine learning algorithm.

The dataset for this project consists of various features regarding a person's lifestyle. We train various models which we compare, analyze and summarize in our report.

**Introduction:**

With the improvement of technology, humans are moving towards a sedentary lifestyle. The more the technology improves, more people will stop doing physical work, this change brings a lot of adverse health effects, for e.g. Strokes. So, in this project, we are determining what factors of one’s lifestyle increase the chances of stroke with the help of machine learning algorithms.

## **Dataset:**

The data consists of 5110 rows.

The data consists of 12 columns named as follows:

| id             | work_type          |
|----------------|--------------------|
| gender         | Residence_type     |
| age            | avg_glucose_level  |
| hypertension   | bmi                |
| heart_disease  | smoking_status     |
| ever_married   | stroke             |

The link to the dataset is given as follows: [link](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

# **Methodology:**

## **Overview:**

### **Preprocessing of the data:**

The data consisted of 201 NaN values in column BMI. These were dropped. Further the data consisted of some string type data which was encoded.

With all encoded data, a correlation heat map was created and according to the heat map, 1 column namely: ID was dropped.

Out of numerous classification algorithms, we implemented the following:

1.  Random Forest Classifier

    Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression.It uses several decision trees and aggregates the result from each tree to predict the final output.

2.  Light GBM boost

    LightGBM is a gradient boosting framework based on decision trees to increases the efficiency of the model and reduces memory usage.

3.  Linear Support Vector Machine

    Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

4.  XGboost classifier

    XGBoost is a popular and efficient open-source implementation of the gradient boosted trees algorithm.

5.  Decision Tree Classifier

    A decision tree is a type of supervised machine learning used to categorize or make predictions based using a tree structure with nodes and leaves at every level.

The number of 1’s is 209 and number of 0’s is 4700 ; i.e. the data is very skewed

![](media/831ec3cc0d570034e749086841532812.png)

Thus any model applied on this dataset may give a good accuracy score but the recall score for 1 will be very less. Thus to improve the recall score of the model we need to oversample 1’s and undersample 0’s

![](media/e085b7854e48f2fe13a31c39200869a0.png)

# **Tabular Comparisons:**

Accuracy is the percentage of correct predictions in total.

Precision is the ratio between predictions that are actually true and all the predictions that are predicted as true.

Recall is the ratio between predictions that are actually true and all the true cases.

F1 score is the harmonic mean between Recall and Precision.

| **Model**               | Accuracy | F1 score | Recall |
|-------------------------|----------|----------|--------|
| **Random Forest**       | 96.06    | 0.03     | 0.02   |
| **LGBM**                | 95.32    | 0.1      | 0.07   |
| **Linear SVC**          | 96.26    | 0        | 0      |
| **Logistic Regression** | 96.33    | 0.04     | 0.02   |
| **XGBoost**             | 94.7     | 0.13     | 0.11   |
| **Decision Tree**       | 91.79    | 0.12     | 0.15   |

oversampling+undersampling

| **Model**               | Accuracy | F1 score | Recall |
|-------------------------|----------|----------|--------|
| **Random Forest**       | 95.18    | 0.15     | 0.11   |
| **LGBM**                | 94.91    | 0.21     | 0.19   |
| **Linear SVC**          | 93.82    | 0.22     | 0.24   |
| **Logistic Regression** | 93.34    | 0.26     | 0.31   |
| **XGBoost**             | 94.16    | 0.17     | 0.16   |
| **Decision Tree**       | 90.29    | 0.11     | 0.16   |

**Changing the Threshold**

This was a key step in our project. Seeing that our best model(Logistic Regression) performed exceptionally well in terms of Recall but lacked in F1 score indicated it had low precision.Since it was our best model yet we decided to use predict proba function to change the threshold at which model predicts stroke(1) or not(0).We made a graph between scores(Recall and F1) and threshold, then from the graph (and data) we chose a threshold whose recall would give good result and at the same time would not perform badly in terms of F1 score.Hence we set our threshold at **65%** meaning if our model predicts a probability of getting a stroke greater than 65% then and only then we would predict stroke=1 unlike the conventional case of predicting stroke=1 for probability greater than 50%.

Initially Accuracy =73.21 ,F1 score=0.21 ,Recall =0.79

Finally Accuracy =81.16 ,F1 score=0.23 ,Recall =0.65

![](media/1ceebe1404b359b95138a89c39c04eaf.png)

**Note:**

Oversampling the test data along with the train data shows better results. However, this must not be done as the test data is not meant to be modified using oversampling or undersampling. Doing this leads to better results on the modified test data but, the model will give bad results on any new data as this leads to loss in generalizability of the model.

Here are the final results

Results:

| **Best Models**                                                                            | Accuracy %  | F1 score  | Recall    |
|--------------------------------------------------------------------------------------------|-------------|-----------|-----------|
| **Decision Tree**                                                                          | 91.79       | 0.12      | 0.15      |
| **Logistic Regression with oversampling of TRAINING data**                                 | 73.21 81.16 | 0.21 0.23 | 0.79 0.63 |
| **Logistic Regression with tampering of TESTING data (NOTE: not used for final pipeline)** | 78.16       | 0.79      | 0.83      |

As we can see, by oversampling the test set the recall improves drastically, but the accuracy decreases.

Hence we use oversampling on **training** data and leave the testing data untouched

# **Conclusion and Analysis:**

The best outcome was provided by the Logistic Regression model. In this project our main focus was to improve the recall score as much as possible without losing accuracy or F1 score.

This is done as the project is required to predict strokes, which is a very critical health condition. We cannot let the model overlook any of the possible cases of strokes, therefore rather than the accuracy for classification, we required a higher recall for the data.

**The final recall for 1 is 0.63**

**The final recall for 0 is 0.82**

**The final accuracy is 81.16%**

# **Contributions:**

Hriday: Implementing Logistic regression,XGboost, Report , Over and undersampling of the data

Rachit: Implementing LGBM,Random Forest, Report,model tuning and threshold

Rohit: Implementing linear SVC,Decision tree,Report, pipeline
