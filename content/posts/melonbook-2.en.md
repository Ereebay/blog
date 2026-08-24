---
title: "Watermelon Book Reading Notes 2. Model Evaluation and Selection"
date: 2020-06-24T16:06:16+08:00
draft: false
math: true
categories: [notes]
tags: [watermelon book]
---
# Model Evaluation and Selection

## Empirical Error and Overfitting

Error rate: the overall proportion of misclassified samples

Accuracy: the proportion of correctly classified samples

Error: the difference between the actual output and the true output; on training samples it is the empirical error or training error, and on new samples it is the generalization error

Overfitting: generalization performance degrades

Underfitting: performs poorly even on the training samples
<!-- more -->
## Evaluation Methods

A testing set is typically used to measure a model's performance, with the testing error serving as an approximation of the generalization error

### Hold-Out Method

Split the dataset D into two mutually exclusive sets, one part as the training set S and the other as the testing set T. When S is large and T is small, the evaluation may not be accurate enough. If T is made larger, then the model trained on S may differ too much from the model that would be trained on D, and the evaluation result will not be accurate enough either.

### Cross-Validation

Split the dataset D into k mutually exclusive subsets, use k-1 of them as the training set and the remaining one as the testing set; this yields k groups of data, and the average of the testing results of these k groups is returned as the final result. The extreme case is leave-one-out, which gives more accurate results but at a greater computational cost.

### Bootstrap

Use bootstrap sampling to draw a new dataset $D'$ with replacement. Train on this new dataset and test on the rest. Advantage: this method is useful when the dataset is small and it is difficult to split training/testing sets effectively. Disadvantage: it changes the distribution of the initial dataset and introduces estimation bias.

### Parameter Tuning and the Final Model

Algorithm parameters have a great impact on the algorithm's performance. Besides choosing an algorithm, its parameters also need to be adjusted, i.e., parameter tuning. After the model has been evaluated and selected using a validation set and the model and parameter configuration are determined, the model must be retrained once on the entire dataset D before testing — this is the finally submitted model. To distinguish it, the data used during model evaluation is called the validation set. The validation set is used to evaluate algorithm selection and parameter tuning, while the testing set is used to evaluate the generalization performance of the algorithm.

## Performance Measures

Performance measure: the evaluation criterion used to measure a model's generalization ability

The most commonly used performance measure for regression tasks is MSE, the mean squared error:

$$E(f ; D)=\frac{1}{m} \sum_{i=1}^{m}\left(f\left(x_{i}\right)-y_{i}\right)^{2}$$

More generally, it can be written as:

$$E(f ; D)=\int_{x \sim D}(f(x)-y)^{2} p(x) d x$$

### Error Rate and Accuracy

The two most commonly used performance measures: error rate and accuracy

Error rate: $E(f ; D)=\frac{1}{m} \sum_{i=1}^{m} I\left(f\left(x_{i}\right) \neq y_{i}\right)$

Accuracy: $\begin{aligned} \operatorname{acc}(f ; D) &=\frac{1}{m} \sum_{i=1}^{m} \mathbb{I}\left(f\left(x_{i}\right)=y_{i}\right) \\ &=1-E(f ; D) \end{aligned}$

### Precision, Recall, and F1

In information retrieval, one usually cares about "how much of the retrieved information the user is interested in" and "how much of what the user is interested in has been retrieved". This motivates the concepts of precision and recall.

For binary classification problems,

Precision = true positives / (true positives + false positives)

Recall = true positives / (true positives + false negatives)

These two metrics usually conflict with each other: when recall is high, precision is relatively low. For example, to select as many good melons as possible, simply selecting every melon guarantees that all the good ones are selected.

P-R curve: the precision-recall curve. If one learner's PR curve can completely envelop another learner's curve, the former learner is better. A more reasonable criterion is the size of the area under the curve, but it is not easy to estimate, so there are three evaluation approaches.

1. Break-Even Point (BEP): the value at which precision equals recall.
2. F1 measure: F1=2*P*R/(P+R)= 2*TP/(total number of examples+TP-TN), based on the harmonic mean of P and R $\frac{1}{F_{1}}=\frac{1}{2}\left(\frac{1}{P}+\frac{1}{R}\right)$
3. The general form of the F1 measure (when P and R are weighted differently): $F_{\beta}=\frac{\left(1+\beta^{2}\right) \times P \times R}{\left(\beta^{2} \times P\right)+R}$, based on the harmonic mean $\frac{1}{F_{\beta}}=\frac{1}{1+\beta^{2}}\left(\frac{1}{P}+\frac{\beta^{2}}{R}\right)$  $\beta$ represents the relative importance of the two: it degenerates to F1 when equal to 1, recall matters more when it is greater than 1, and precision matters more when it is less than 1.

Sometimes P and R need to be computed over n binary confusion matrices and then averaged, giving macro-precision, macro-recall, and macro-F1. Alternatively, the corresponding elements of the confusion matrices can be averaged first to obtain averaged positive and negative counts, from which micro-precision, micro-recall, and micro-F1 are then computed.

### ROC and AUC

ROC: Receiver Operating Characteristic curve.

A learner typically outputs a probability prediction for an input sample, which is then compared against a classification threshold. The model ranks the samples most likely to be positive at the front and the least likely at the back, then splits the samples into two parts at a certain cut point: the first part is judged positive and the second part negative.

Different classification tasks adopt different cut points: if precision matters more, the cut point is moved forward; if recall is valued more, it is moved backward.

Sort the samples according to the learner's prediction results, and compute two values in that order.

Vertical axis: True Positive Rate (TPR): $T P R=\frac{T P}{T P+F N}$

Horizontal axis: False Positive Rate (FPR): $FPR=\frac {FP} {TN+FP}$
![Untitled](http://cdn.ereebay.me/hexo/Untitled.png)

When comparing learners, if one curve can envelop the other, the former is better; if they cross, neither clearly dominates. A more reasonable judgment is to compare the area under the ROC curve, i.e., the AUC.

$$A U C=\frac{1}{2} \sum_{i=1}^{m-1}\left(x_{i+1}-x_{i}\right)\left(y_{i}+y_{i+1}\right)$$

Loss: $\operatorname{l_{rank}}=\frac{1}{m^{+} m^{-}} \sum_{x^{+} \in D^{+}} \sum_{x \in D^{-}}\left(\mathbb{I}\left(f\left(x^{+}\right)<f\left(x^{-}\right)\right)+\frac{1}{2} \mathbb{I}\left(f\left(x^{+}\right)=f\left(x^{-}\right)\right)\right)$

AUC= 1-lrank

[https://datawhalechina.github.io/pumpkin-book/#/chapter2/chapter2](https://datawhalechina.github.io/pumpkin-book/#/chapter2/chapter2) For more details, see the Pumpkin Book

### Cost-Sensitive Error Rate and Cost Curve

To weigh the different losses caused by different types of errors, errors can be assigned "unequal costs"

Taking binary classification as an example, set up a cost matrix

Binary classification cost matrix
|  True class| Predicted class 0 |Predicted class 1|
|  ----  | ---|---|
| 0  | 0|cost01|
| 1 | cost10| 0

If misclassifying 0 as 1 is considered the greater loss, then cost01 is greater than cost10; the greater the difference in loss, the greater the difference between the values.

The performance measures introduced above all implicitly assume equal costs; under unequal costs, the goal becomes minimizing the total cost overall

The cost-sensitive error rate is:

$$
E(f ; D ; \cos t)=\frac{1}{m}\left(\sum_{x_{i} \in D^{+}} \mathbb{I}\left(f\left(x_{i}\right) \neq y_{i}\right) \times cost_{01}, \quad+\sum_{X_{i} \in D^{-}} \mathbb{I}\left(f\left(x_{i}\right) \neq y_{i}\right) \times cost_{1 0}\right)
$$

Under unequal costs, the ROC curve cannot reflect the learner's expected total cost, but the cost curve can serve this purpose.

The horizontal axis is the positive-class probability cost over [0,1]:

$$
P(+) cost=\frac{p \times cost_{01}}{p \times cost_{01}+(1-p) \times cost_{10}}
$$

where p is the probability that an example is positive

The vertical axis is the normalized cost over [0,1]:

$$
cost_{\text {norm }}=\frac{ F N R \times p \times cost_{01}+FP R \times(1-p) \times \cos t_{10}}{p \times cost_{01} +(1-p) \times cost_{10}}
$$

FNR is the false negative rate, and FPR is the false positive rate. FNR=1-TPR

## Comparative Tests

Performance comparison in machine learning usually relies on statistical hypothesis tests, for the following reasons:

1. We want to compare generalization performance, but what experiments evaluate is performance on the testing set, and the two comparison results do not necessarily agree
2. Performance on the testing set depends heavily on the choice of testing set
3. The algorithm itself involves a certain degree of randomness

### Hypothesis Testing

The error rate ${\epsilon}$ serves as the performance measure.
For a learner with error rate ${\epsilon}$, suppose we obtain a testing error of $\hat{\epsilon}$ on the testing set; this means that $\hat{\epsilon} \times m$ of the $m$ samples are misclassified. It is easy to derive that the probability of obtaining testing error $\hat{\epsilon}$ on a test set of $m$ samples, for a learner with generalization error rate ${\epsilon}$, is:

$$
P(\hat{\epsilon} ; \epsilon)=\left(\begin{array}{c}m \\ \hat{\epsilon} \times m\end{array}\right) \epsilon^{\hat{\epsilon} \times m}(1-\epsilon)^{m-\hat{\epsilon} \times m}
$$

A binomial distribution is used for the hypothesis test. The hypothesis takes the form $\epsilon \leq \epsilon_{0}$, and the maximum error rate we can observe with probability $1-\alpha$ is:

$$
\bar{\epsilon}=\max \epsilon \quad \text { s.t. } \sum_{i=\epsilon_{0} \times m+1}^{m}\left(\begin{array}{c}m \\ i\end{array}\right) \epsilon^{i}(1-\epsilon)^{m-i}<\alpha
$$

If the testing error $\hat{\epsilon}$ is smaller than the critical value $\bar{\epsilon}$, we accept the hypothesis at the $1-\alpha$ confidence level; otherwise, at the significance level $\alpha$, we conclude that the generalization error rate is greater than $\epsilon$.

If the hold-out method or cross-validation is applied multiple times, k testing error rates are obtained. The mean and variance can then be computed, and a t-test can be used for the hypothesis test. The hypothesis in this case is that the mean equals the error rate. The variable:

$$
\tau_{t}=\frac{\sqrt{k}\left(\mu-\epsilon_{0}\right)}{\epsilon}
$$

follows a t-distribution with k-1 degrees of freedom.

More test methods are described on p.41 of the Watermelon Book

## Bias and Variance

Besides estimating generalization performance, we also want to understand why the model performs the way it does, which calls for bias-variance decomposition. Taking a regression task as an example, y_D is the label of x on the dataset, y is the true label of x, and f(x,D) is the predicted output.

The expected prediction is then: $\bar{f}(x)=\mathbb{E}_{D}[f(x ; D)]$

Training sets with different numbers of samples give rise to variance. Then perform the decomposition:

$$
\begin{aligned} E(f ; D) &=\mathbb{E}_{-} D\left[\left(f(x ; D)-y_{D}\right)^{2}\right] \\ &=\mathbb{E}_{-} D\left[(f(x ; D)-\bar{f}(x))^{2}\right]+(\bar{f}(x)-y)^{2}+\mathbb{E}_{-} D\left[\left(y_{D}-y\right)^{2}\right] \\ &=\operatorname{bias}^{2}(x)+\operatorname{var}(x)+\varepsilon^{2} \end{aligned}
$$

The generalization error can be decomposed into the sum of bias, variance, and noise

- Bias measures the deviation of the algorithm's expected prediction from the true result, characterizing the fitting capability of the learning algorithm itself
- Variance measures the change in learning performance caused by changes in training sets of the same size, characterizing the impact of data perturbation
- Noise expresses the lower bound of the expected generalization error that any learning algorithm can achieve on the current task, characterizing the difficulty of the learning problem itself.
