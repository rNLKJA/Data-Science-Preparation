# Machine Learning 

This section covers concepts, techniques, and applications of machine learning for interviews, projects, and research.

## Table of Content

- [Machine Learning](#machine-learning)
  - [Table of Content](#table-of-content)
  - [Machine Learning Breath](#machine-learning-breath)
    - [What is Bias-Variance Tradeoff?](#what-is-bias-variance-tradeoff)
    - [What is cross-validation?](#what-is-cross-validation)
      - [CV AUC is 0.90. However, when the model is productionsed, AUC drops to 0.75. Why?](#cv-auc-is-090-however-when-the-model-is-productionsed-auc-drops-to-075-why)
      - [What happens if you increase the number of folds?](#what-happens-if-you-increase-the-number-of-folds)
      - [Does cross-validation improve model performance?](#does-cross-validation-improve-model-performance)
    - [How would you handle multicollinearity?](#how-would-you-handle-multicollinearity)
      - [Suppose that the feature set contains 800 variables in a supervised model. How would you handle multicollinearity?](#suppose-that-the-feature-set-contains-800-variables-in-a-supervised-model-how-would-you-handle-multicollinearity)
    - [Imbalanced Labels](#imbalanced-labels)
      - [Best Practice #1: Choose the right metric for evaluating model performance.](#best-practice-1-choose-the-right-metric-for-evaluating-model-performance)
      - [Best Practice #2: Conduct re-sampling of miority and majority class](#best-practice-2-conduct-re-sampling-of-miority-and-majority-class)
      - [Best Practice #3: Try using cost-sensitive learning](#best-practice-3-try-using-cost-sensitive-learning)
    - [What is the curse of dimensionality? How do you prevent it?](#what-is-the-curse-of-dimensionality-how-do-you-prevent-it)
      - [How to you mitigate curse of dimensionality?](#how-to-you-mitigate-curse-of-dimensionality)
    - [What is AUC? How is it helpful when labels are imbalanced?](#what-is-auc-how-is-it-helpful-when-labels-are-imbalanced)
      - [AUC (Area Under the Curve)](#auc-area-under-the-curve)
      - [ROC Curve (Receiver Operating Charateristic Curve)](#roc-curve-receiver-operating-charateristic-curve)
  - [Machine Learning Algorithm](#machine-learning-algorithm)
  - [Deep Learning](#deep-learning)
  - [Machine Learning Prediction](#machine-learning-prediction)

## Machine Learning Breath

### What is Bias-Variance Tradeoff?

The variance-bias trade-off is a fundamental concept in machine learning, which assesses how well a model has fitted the data for predictions. There are two properties - variance and bias - that assess model fits. And, there's a trade-off between the two such that an increase in one decreases the other. And, the main for machine learning algorithms and various techniques (e.g. ensembling and regularization) is to find the optimal balance between the two.

Let's consider the meaning of bias and variance:
- **Bias**: is the difference between the average prediction of your model and the true value you are trying to predict. High bias means your model is overly simplistic. It has strong assumptions about the data which prevents it from learning the real underlying patterns. This leads to underfitting.
- **Variance**: measures how much your model's prediction change with different training datasets. High variance means the model is highly sensitive to the specific data it's trained on, capturing noise and peculiarities rather than the generlisable trend. This leads to overfitting.

**The Trade-off**
- **Complex Models**: flexible models (like polynomial linear models, deep neural networks, decision trees with many splits) have the capabcity to fit complex patterns in the data. This is reduces bias but can lead to high variance if they start fitting the noise in the training data rather than the true underlying trend.
- **Simple Models**: linear models or shallow decision trees have bias because of their simplicity assumptions. However, they are less prone to overfitting and tend to have lower variance.

**Tehcniques to Address the Trade-off**
- **Regularisation**: adds penalities to complex models to favor simpler explanations (e.g. L1/L2 regularsation).
- **Hyper-parameter Tuning using Cross-Validation**: Helps evaluate the best combination of parameters from multiple splits of data and produce the optimal complexity in decision boundary that minimise both variance and bias.
- **Ensembling Methods**: Combines multiple models to reduce variance (e.g., bagging, random forests).

### What is cross-validation? 

Suppose that you randomly sample and allocate 70% of your data for training the model and the remaining 30% of the data for testing your model. Let's also assume that the prediction problem is regression so mean square error (MSE) is used to evaluate the model. You compute the MSE of the model prediction on testing data and conclude 678.34.

This approach seems to work, but there is a drawback. Suppose that model performance is evaluated on new testing data, resulting in 824.44 MSE. This discrepency suggests that there is variability in the model prediction error from one set of data to the next. Cross-validation reduces this uncertainty in estimating the error by averaging multiple testing errors across multiple folds of the data.

This procedure of cross-validation is simple. You choose K number of folds, which partitions the entire data into K folds, as shown below.

<img src="https://files.cdn.thinkific.com/file_uploads/481328/images/1a3/756/17f/1621484167709.jpg?width=1920&dpr=2" align="center" width=300 />

Each fold is subject to validation while the rest become data used for training the model. This process is repeated K times to ensure that all of the folds of the data are computed with errors.

Finally, the errors are averaged to produce a single error score with reduced uncertainty.

#### CV AUC is 0.90. However, when the model is productionsed, AUC drops to 0.75. Why?

The discrepency in a model performance offline vs. online happens when a holdout test wasn't utilised to measure the model performance. Cross-validation will work well when the observations in the data are independent, meaning that the past observation does not influence the future. However, in many modelling exercises, observations are usually autocorrelated, i.e.:
1. a fraud user previously banned will re-appear with a new account and new behaviors to avoid detection
2. an online shopper will change spending patterns over time

Given the dependence, you can see how it's problematic to use future data to predict and evaluate past data in cross-validation. Here, the better design is to utilised time-series cross-validation always training on historical data and predicting and evaluating future data.

#### What happens if you increase the number of folds?

If you increase the number of folds, that's more values you are using the average. This will decrease the variance of the evaluation whill increasing the bias of the evaluation.

#### Does cross-validation improve model performance?

No,cross-validation is not designed to improve model performance. It's designed to improve the measurement of model performance.

### How would you handle multicollinearity?

To begin answering this question, let's first provide explanation of the definition of multicolinearity. Multicollinearity is the presence of two or more correlated features in a model. The best practice in building a high-predictive, interpretable machine learning models is to remove multicollinearity.

Multicollinearity can harm:
1. The predictve performance of a model because of overfitting
2. The interpretability of the model such that the variable importance of a feature correlating which author would be inaccurate
3. The maintainability of large correlated features in a production environment

Now, let's address the interviewer's question to treating multicollinearity in a model. You can list techniques:
1. Use Pearson and Spearman correlations to identify correlated variables. Use Pearson correlation of the relationship between two variables is linear. If not, use Spearman.
2. Employ variance inflation factor (VIC) to identify correlated variables in a regression model
3. Apply the wrapper method such as backward, forward, or stepwise to build a model that uses a feature set with a low presense of multicollinearity
4. Use regularised regression such as elastic net, lasso, or ridge. You can use the constructed model as the final model for prediction, or extract features with nonzero coefficients as de-correlated features for a final mdoel such as a GBM (Gradient Boosting Machine) or random forest or neural network.
5. Use principal component analysis (PCA) to compress a feature set into a smaller set of de-correlated features.

#### Suppose that the feature set contains 800 variables in a supervised model. How would you handle multicollinearity?

Handling multicollinearity is a combination of art and best practice. There are many approahces. Here is one approach that could work. Assume that the 800 variables further break down into 300 categroical variables and 500 numerical variables. To make de-corrleation easy, transformation the 300 categorical variables into numerical variables with numerical encodings such as weight-of-evidence (WOE), mutual information or class probability. Prior to computing correlations on pairs of any two variables, scale to remove outliers and standardise the numerical range.

### Imbalanced Labels

Suppose you are building a credit fraud model with the miniority class being less than 1%. How would you build a classification model that can handle an extremently imbalanced dataset.

**Solution**

Always relate back to the problem which is credit fraud. Just simply listing techniques for handling imbalanced datasets is not enough.

This solution will cover popular tecniques and lightly touch the theory behind how each techniques works. There are depths of statisical underpinning on why the techniques work and when those fail, but this guide should provide a guide on how to respond to the interwer's question.

Common techniques include:
1. choosing the right criterion to measure model performance (e.g. AUC, F1 score, precision, recall)
2. resampling techniques to balance the class (e.g. oversampling, undersampling, SMOTE)
3. applying cost-sensitive learning to penalise the model for misclassifying the minority class

Before exploring each technique, let's add sturcture to the context that the interviwer posed to you. This further demonstrates to the interviewer that you have a framework on how you approach the problem. You can say something along the lines of: "I'm going to assume that I have access to historical data with records from 2015 to 2017. Let's also assume that there are 2.4 million credict cart transactions, and about 0.5% are known fraudulent transactions. That's merely 12,000 known bad compared to millions of known goods in the dataset."

#### Best Practice #1: Choose the right metric for evaluating model performance.

Do not use accuracy, which is 

$$
\text{Accuracy} = \cfrac{\text{True Positive} + \text{True Negative}}{\text{True Positive} + \text{True Negative} + \text{False Positive} + \text{False Negative}}
$$

The class distribution is heavily skewed toward the good population (negatives). When your model yields the following result below, based on accuracy, the model performance is 95%.

| | True | False |
| -- | -- | -- |
| Pred True | 1,200 | 0 |
| Pred False | 10,800 | 2,280,000 | 

$$
\text{Accuracy} = \cfrac{1,200 + 2,280,000}{1,200 + 0 + 10,800 + 2,280,000} = 95\%
$$

For a model that only predicts 1,200 of the 12,000 acutal bads, or 10% correctly, the accuracy at 95% makes it appear that the model is doing well. But, in reality, it is not.

**Do not use use ROC-Curve**

When the class distribution is extremely skewed toward goods over bads, ROC-Curve becomes ineffective in evaluating the performance of classification model.

Consider that ROC-curve is a plot formed by true-positive rates (TPR) and false-positive rates (FPR) across the threshold range from 0 to 1, inclusive. The area under the curve (AUC) is the metric used to evaluate the model performance. The AUC ranges from 0 to 1, where 0.5 is random guessing and 1 is perfect prediction.

$$
\text{TPR} = \cfrac{\text{True Positive}}{\text{True Positive}+\text{False Negative}},\ \text{FPR} = \cfrac{\text{False Positive}}{\text{True Negative}+\text{False Positve}}
$$

Consider a simple example below which consists of 10,000 observations with 9990 goods and 10 bads as shwon below. The decile represents the probability threshold applied to testing data. The size represents the total number of observations with probability scores above or equal to the threshold. At each threshold, the corresponding true-positive (TP), false-negative (FN), true-negative(TN), false-positive (FP), TPR, FPR are computed.

<img src="https://files.cdn.thinkific.com/file_uploads/481328/images/150/32c/e98/1621484921943.jpg?width=1920&dpr=2" align="center" />

Numbercial Summary:

<img src="https://files.cdn.thinkific.com/file_uploads/481328/images/bd5/64a/3e9/1621484922280.jpg?width=1920&dpr=2" align="center" />

When you observe the numerical summary, the poor prediction of true positives is overshadowed by the extremely disproporitionate number of true negatives. Consider that the majority of the true=negatives mass on the lower range of probability scores than that of the true-positives.

When you examine, decile threshold at let's say, 0.6, TPR is 0.8 and FPR is 0.3003. When you examine the model on this threshold, the model seems to do quite well given that 80% of the bads will be predicted accurately and about 30% of the negatives will misclassified as good. However, this overlooks the volumen of the total negatives misclassified in relation to the true positives. In other words, this metric is missing precision.

**Use PR-Curve**

The best practice is to evaluate the model using PR-Curve, which uses precision and recall, evaluated across model score range from 0 to 1, inclusive. Precision and recall both target true-positive as a measure for model performance.

$$
\text{Recall} = \cfrac{\text{True Positive}}{\text{True Positive} + \text{False Negative}},\ \text{Precision} = \cfrac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}
$$

Note that recall is synonymous with TPR. Precision, on the other hand, includes false-positives - which is the key to evaluating a model under an extremly imbalanced problem. Reviewing threshold 0.6 in the numerical summary above, you observe that precision is merely 0.27%. This is a far different picture of the model's ability to predict positives. Even the curve provides a different picture than ROC-curve:

<img src="https://files.cdn.thinkific.com/file_uploads/481328/images/044/5c3/4c1/1621484921840.jpg?width=1920&dpr=2" align='center' />

#### Best Practice #2: Conduct re-sampling of miority and majority class

There are three widely-known variances of re-sampling techniques - downsampling, oversampling, and SMOTE, We will cover downsampling and oversampling in this solution. For SMOTE, there is plenty of academic literature that covers the technique in depth.

The intuition behind downsampling and oversamping is simple. The majority class dispered across the areas of minority class causes difficulty in any classification model to drwa a decision boundary that separates the two classes.

<img src="https://files.cdn.thinkific.com/file_uploads/481328/images/d86/aa5/f34/1621484922493.jpg" align='center' width=500 />

When there's too much noise from the majority class (orange) around the miniority class (blue) as shown in the illustration on the left above, the boundary between the orange and blue data points becomes fuzzy. Down-sampling can reduce the noise; thereby, help a classification model form a better boundary as shown on the right.

With a similarity objective in mind, oversampling of miniority class can also be performed. Essentially, this is boostrapping the minority class to improve the balance between the ratio of goods and bads.

#### Best Practice #3: Try using cost-sensitive learning

In credit fraud modelling, there are two types of decisions and four types of outcomes. In terms of decisions, the model labels them as fraud or non-fraud. But, based on the actual label, there are four outcomes each associated with its own cost:

| | Fraud | Non-Fraud |
| -- | -- | -- |
| Pred Fraud | c(1,1) | c(1,0) |
| Pred Non-Fraud | c(0,1) | c(0,0) |

There is a multitude of cost-sensitive learning. One main type you should be aware of is cost-sensitive learning with respect to threshold determination.

When class is highly imbalanced, you never want to choose 0.5 as the threshold for predicting class as fraud. Some measure incorporating cost is required.

C(1,0) is the cost of false positive (FP) while C(0,1) is the cost of false negative (FN). You can determine your threshold, P*, based on the following:

$$
\cfrac{\text{False Positve}}{\text{False Negative} + \text{False Positive}} = P^*
$$

Predict class as fraud if $\Pr(\text{Fraud} | X) \ge P^*$.

The derivation of the formula is out of scope in the solution. There is plenty of literature that covers the topic of what is an optimal model threshold based on costs. Given that not all fraud problems are the same and the cost of each determination is different from company to company, cost-sensitive classification is subject to hange.

### What is the curse of dimensionality? How do you prevent it?

The curse of dimensionality refers to a set of challenges and problems that arise when working with high-dimensionality data (datasets with many features). The core issue is the sparsity:

As the number of dimensions increases, the volumne of the data space increases exponentially. This means data points become increasingly scattered and far apart, making it difficult to find patterns.

As the data points become less clustered and sparse, the decision boundary begins to overfit, which decreases generalistion. Additionally, curse dimensionality, unnecessarily increases model training and inference time.

#### How to you mitigate curse of dimensionality?

The following methods can handle curse of dimensionality:
- **Dimensionality Reduction**: Techniques like PCA to project data into lower dimensions
- **Features Selection**: Identify the most important features and dropping the rest
- **Reguarlisation Parameters**: Every common ML algorihtms contain parameters that mitigate against overfitting.
  - Decision Tree-Pruning
  - Random Forest - Bootstrap, Number of Trees, Column and Row Sample
  - XGBoost - Bootstrap, Column and Row Sample, L1/L2 Regularisation Term
  - Neural Network - Dropout, L1/L2 Regularisation Term 

### What is AUC? How is it helpful when labels are imbalanced?

Suppose there are a total of 100 observations - 99 goods and 1 bad. Your classification model predicts all 100 observations as good, resulting in 99% accuracy. The model is great at correctly classifying goods with a 100% true positive rate, but horrendous at classifying the bad with 100% false positive rate.

<img src="https://files.cdn.thinkific.com/file_uploads/481328/images/6b9/bdf/c46/download-2.png?width=1920&dpr=2" width=500 align=center />

AUC is the metric to apply in such a case when the labels are imbalanced. AUC stands for "Area Under the Curve", and it is typically used in the context of the ROC curve, or Receiver Operating Characteristics curve, in statistics and machine learning. Here's a breakdown of each term:

#### AUC (Area Under the Curve)

1. **Overview**: AUC refers the area under a curve in a graph. In the context of classification problems in machine learning and statistics, it usually refers to the area under the ROC curve, which is a graphical representation of a model's diagnostic ability.
2. **Significance**: AUC provides a single scalar value that represents the likelihood that the classifer will rank a randomly chosen positive instance higher than a randomly chosen negative one. It's often used to evalute the performance of binary classication algorithms, though it can be extended to multi-class classification.
3. **Range**: AUC ranges from 0 to 1, with a value of 05 representing a model that performs no better than random and a value of 1 representing a perfect model. Generally, an AUC above 0.7 is considered acceptable, but this threshold may vary dpeending on the application.

#### ROC Curve (Receiver Operating Charateristic Curve)

1. **Plot**: The ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier as its discrimination threshold is varied. It is created by plotting the true positive rate (TPR, or sensitivity) against the false positive rate (FPR, or 1-specificity)
2. **Interpretation**: Each point on the ROC curve represents a different threshold used to convert the model's real-valued predictions into binary classifications. The cruve illustrates the trade-off between sensitivity and specificity at various thresholds
3. **Usage**: It's a common used tool for evaluating the performance of classification algorithms, especially in the field of medical decision-making, and for comparing different classifiers

## Machine Learning Algorithm

## Deep Learning

## Machine Learning Prediction