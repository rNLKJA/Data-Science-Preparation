# Machine Learning 

This section covers concepts, techniques, and applications of machine learning for interviews, projects, and research.

## Table of Content

- [Machine Learning](#machine-learning)
  - [Table of Content](#table-of-content)
  - [Machine Learning Breath](#machine-learning-breath)
    - [What is Bias-Variance Tradeoff?](#what-is-bias-variance-tradeoff)
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

## Machine Learning Algorithm

## Deep Learning

## Machine Learning Prediction