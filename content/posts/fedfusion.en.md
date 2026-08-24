---
title: "[Paper] TOWARDS FASTER AND BETTER FEDERATED LEARNING: A FEATURE FUSION APPROACH Reading Notes"
date: 2020-06-24T18:55:03+08:00
draft: false
math: true
categories: [notes]
tags: [paper, federated learning]
---
# TOWARDS FASTER AND BETTER FEDERATED LEARNING: A FEATURE FUSION APPROACH

## Abstract

This paper mainly proposes a feature fusion approach to accelerate and improve the performance of federated learning.

<!-- more -->

## Introduction

Nowadays many smart devices rely on pretrained models, which makes the machine's inference capability lack personalization and flexibility. At the same time, smart terminals also generate a large amount of valid privacy-sensitive data, which can improve the personalization capability of these models. Federated learning, a distributed training algorithm that can train models directly on terminals, solves this problem. Among federated learning algorithms, those represented by FedAvg effectively alleviate the privacy issues in information exchange, but later research has also shown that federated learning still has problems such as: **computational consumption**, **model accuracy**.

This paper proposes a feature-fusion federated learning algorithm, FedFusion, which fuses the features of the global model and the local model. The three main contributions of this paper: 1. introducing a feature fusion mechanism; 2. fusing the features of the local model and the global model in an effective and personalized way; 3. experiments show that the model outperforms the baselines in both accuracy and generalization ability while reducing communication by more than 60%.

## Related Work

Mainly the FedAvg algorithm of Federated Learning; not much to say here.

## Methods

It is mainly divided into the feature fusion module and the FedFusion algorithm.

### Feature Fusion Modules

![fedfusion1](http://cdn.ereebay.me/hexo/fedfusion1.png)

The blue features in the figure are the two-channel features extracted by the local model, and the gray ones are the two-channel features extracted by the global model. The figure shows three feature fusion methods: Conv, Multi, Single.

Conv:

$$F_{c o n v}\left(E_{l}(x), E_{g}(x)\right)=W_{c o n v}\left(E_{g}(x) \| E_{l}(x)\right)$$

where $W_{c o n v}$ denotes a learnable weight matrix of shape 2C*C. The concrete operation is to concatenate the global and local features and then apply a convolution.

Multi:

$$F_{m u l t i}\left(E_{l}(x), E_{g}(x)\right)=\lambda E_{g}(x)+(1-\lambda) E_{l}(x)$$

The multiplication operation uses a lambda weight matrix to perform a weighted sum of local and global.

Single:

$$F_{\text {single}}\left(E_{l}(x), E_{g}(x)\right)=\lambda E_{g}(x)+(1-\lambda) E_{l}(x)$$

The addition operation uses a scalar lambda weight to perform a weighted sum of local and global.

### FedFusion

![fedfusion2](http://cdn.ereebay.me/hexo/fedfusion2.png)

The training procedure uses the features of the previous round's global model to participate in this round's model feature aggregation training.

## Experiment

### Experiment setup

Datasets: Mnist, Cifar10

Data partition methods:

1. An artificial non-IID partition, where each node contains only two classes
2. A user-specific non-IID partition, where each node contains similar classes but with different distributions, similar to multi-task learning
3. IID distribution

### Artificial Non-IID Partition

![fedfusion3](http://cdn.ereebay.me/hexo/fedfusion3.png)

Experimental results of two random runs of the artificial non-IID sampling scheme. The experimental results show that the multi fusion method works best; the conv fusion method converges slightly faster, but its final result is not as good as multi; none and single are both mediocre.

The paper's explanation is that the multi operation allows the model to select the feature maps that are effective for local data for fusion, while the single operation is a scalar and cannot select specific channels of the feature maps.

### User-Specific Non-IID Partition

![fedfusion4](http://cdn.ereebay.me/hexo/fedfusion4.png)

In terms of accuracy, FedFusion is much higher than FedAvg, with conv converging faster and also achieving higher accuracy.

![fedfusion5](http://cdn.ereebay.me/hexo/fedfusion5.png)

The figure above shows the degree of communication reduction of FedFusion compared with FedAvg. From the results, in the user-specific non-IID scenario, the conv fusion method works better. This is because, under the user-specific non-IID partition, the classes of the data are similar, only with different distributions. And the conv fusion method is stronger at integrating the feature maps from the local and global models — that is, the knowledge of the data distributions of different nodes.

![fedfusion6](http://cdn.ereebay.me/hexo/fedfusion6.png)

For the impact on generalization ability: when a new node joins, FedFusion only needs about 60 local epochs to fit, having a better initialization than the other methods.

### IID Partition

The authors believe that the IID distribution also needs to be evaluated, because if a strategy cannot handle the IID distribution, its effectiveness is questionable.

![fedfusion7](http://cdn.ereebay.me/hexo/fedfusion7.png)

The multi and conv fusion methods can achieve better accuracy with the least communication cost. In terms of the final converged accuracy, there is a great improvement compared with the other methods.

To summarize the three fusion methods:

The multi operation mainly makes a more flexible and more interpretable selection between the local feature maps and the global feature maps. Each channel of the weight vector represents the weight of the corresponding channel of the global feature map. When a gap appears between the classes of the data, the multi operation can select the most effective feature maps for fusion. The conv operation is more effective at integrating the knowledge of the global and local models. If the data on the nodes has similar classes but different distributions, the conv fusion method is better. The single fusion method brings a slight improvement.

## Conclusion

Using feature map fusion reduces the communication volume, improves model performance, and also improves the generalization ability for new nodes.
