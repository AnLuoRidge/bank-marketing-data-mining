# Bank Marketing Data Mining
Prediction of the subscription rate of a bank marketing data set.

The data set is from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

# Introduction
This project aims to predict the subscription of the deposit based on provided data set. The data set is from direct marketing campaigns of a Portuguese banking institution.
There are 21 attributes provided which can be divided into the bank client information, campaign data, and social and economic index.
The detail of the attributes is list below.

| Attribute | Description | Type |
| --- | --- | --- |
| age | Age of the client | Ratio |
| job | Client&#39;s occupation | Nominal |
| marital | Marital status | Nominal |
| education | Client&#39;s education level | Nominal |
| default | Indicates whether the client has credit in default | Nominal |
| housing | Indicates whether the client has a housing loan | Nominal |
| loan | Indicates whether the client as a personal loan | Nominal |
| contact | Type of contact communication | Nominal  |
| month | Month that last contact was made | Nominal  |
| day\_of\_week | Day that last contact was made | Nominal  |
| duration | Duration of last contact in seconds | Ratio  |
| campaign | Number of contacts performed during this campaign for this client (including last contact) | Ratio |
| pdays | Number of days since the client was last contacted in a previous campaign | Ratio  |
| previous | Number of contacts performed before this campaign for this client | Ratio |
| poutcome | Outcome of the previous marketing campaign | Nominal  |
| emp.var.rate  | Employment variation rate (quarterly indicator) | Ratio |
| cons.price.idx  | Consumer price index (monthly indicator) | Ratio |
| cons.conf.idx  | Consumer confidence index (monthly indicator) | Ratio |
| euribor3m | Euribor 3-month rate (daily indicator) | Ratio |
| nr.employed | Number of employees (quarterly indicator) | Ratio |
| Final\_Y | The subscription of the deposit | Norminal |

The report records the process of the try and error, and finally provides the best approach to the prediction problem. Sklearn is the main framework used in this project.
# Data cleaning and preprocessing
## Data cleaning
The data cleaning consists of filling of the missing values, fixing of inconsistency, removing of outliers, and reduction of duplication.
### Missing values
There are six attributes contain missing values which are &#39;job&#39;, &#39;marital&#39;, &#39;education&#39;, &#39;default&#39;, &#39;housing&#39;, and &#39;loan&#39;,
Since none of them are numeric type, they are filled by mode. For example, all the missing values in &#39;job&#39; are replace by &#39;blue collar&#39;.
### Inconsistency
According to the attribute description, a value of &#39;999&#39; in the &#39;pdays&#39; means the client is new. Thus, the corresponding &#39;poutcome&#39; would be &#39;nonexistent&#39;. However, some entries are inconsistent with this rule. So, they are removed.
### Outliers
Since most values in the data points are within the normal range, so it is reasonable to train the model without outliers. By doing so, the model should predict well on the normal data points. The range of outlier are identified by the boxplot.
For example, all the duration exceeds 500s are treated as outliers.
### Duplication
Reduce the duplication helps improve the performance of the model.
## Data preprocessing
### One-hot encoding
Since nominal data cannot be used in many classifiers, all the nominal data are encoded by one-hot. This is done by the dummy() function in Pandas.
### Normalisation
To reduce to the complexity of computation and make the weight of the value the same, z-score normalisation is performed.  Z-score normalise the data points in the sphere zone which is better for K-nearest neighbour (KNN).  KNN is the assigned classifier in this project.
### Discretion
Too many kinds of values in attributes consume a lot of computing resource and may downgrade the performance of the model. For example, &#39;age&#39; is discrete into &#39;young&#39;, &#39;mid&#39;, and &#39;elder&#39;.
# Approach
At beginning, a feature ranking will be performed. With the selection of important features, the candidate classifier will be tested one by one. In this project, classifiers used are K-nearest neighbour, decision tree, random forest, and gradient boosting.
With each classifier, the feature selection will be executed first. This time, different combination and number of features are tested. After finding out the best group of features, parameters are tuned by grid search.
The performance of each classifier will be compared by F score because of the imbalanced of this dataset. Comparing with the customer who does not subscribe a deposit, there is fewer customers subscribed. Scoring by accuracy is affected by the accuracy of negative behaviour too much, while F score treated precision and recall in the same weight.
For the same reason, Precision-Recall curve is better than the ROC curve.
When the best classifier is chosen, more actions could be performed to improve the score of the model, such as
1) change the sampling method the training set to reduce the imbalance.
2) predict the missing value with different classifier.
3) adjust on binning strategy.
4) use a new classifier.
5) combine different classifiers.
# Classification
## Ranking of feature importance
Figure 1 Ranking of feature importance
## Classifier selection
### K-nearest neighbour (assigned)
K-nearest neighbour (KNN) is a supervised classifier, and there is only one parameter to tune which is k. KNN does not support the feature selection in sklearn, so this step skipped. The parameter k is tested from 1 – 100, and 1 get the best F score.
Figure 2 F score with different k
Following is the confusion matrix when k is 1.
Figure 3 Confusion matrix of KNN
The F score is 0.458.
### Decision tree
Feature selection is performed first. According to the graph, nine features is the best option.
Figure 4 Different feature numbers in decision tree
The nine features are &#39;age&#39;, &#39;duration&#39;, &#39;cons.conf.idx&#39;, &#39;euribor3m&#39;, &#39;marital\_single&#39;, &#39;education\_university.degree&#39;, &#39;housing\_no&#39;, and &#39;poutcome\_success&#39;.
Figure 5 Confusion matrix of decision tree
The F score is 0.626.
### Random forest
The final F score of random forest is 0.630. The confusion matrix is showed below.
Figure 6 Confusion matrix of random forest
### Gradient Boosting
Figure 7 Confusion matrix of gradient boosting
The F score is 0.636.
### Comparison
| **Classifier** | **F score** |
| --- | --- |
| Gradient boosting | 0.636 |
| Decision tree | 0.626 |
| Random forest | 0.630 |
| KNN | 0.458 |
Table Comparison of F score
## Best classifier – Gradient Boosting
The features selected are:
- age
- duration
- conf.idx
- euribor3m
- marital\_single
- education\_university.degree
- housing\_no
- poutcome\_success
The parameters settings are n\_estimators=113, learning\_rate=0.2, max\_depth=3, subsample=1, criterion: &#39;friedman\_mse&#39;. The rest remains default setting.
The F score is 0.636, and corresponding accuracy is 92.20% (The accuracy on Kaggle is lower than this). The result is based on the split of the local training data set in which 80% is training data and 20% is test data.
