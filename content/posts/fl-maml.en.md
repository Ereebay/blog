---
title: "Improving Federated Learning Personalization via MAML Paper Notes (Incomplete)"
date: 2019-12-02T15:07:30+08:00
draft: false
categories: [notes]
---

# Improving Federated Learning Personalization via MAML

## Abstract

1. FL algorithms share many similarities with MAML and can be interpreted through meta-learning algorithms
2. Fine-tuning gives the global model stronger accuracy while making personalization easier
3. Models trained on standard centralized datasets are harder to personalize than those trained with FedAvg

<!-- more -->

## Introduction

1. Points out the connection between FL and MAML, and interprets the FL algorithm via MAML
2. Improves FedAvg with a two-stage approach of training and fine-tuning
3. Finds that FedAvg is essentially a meta-learning algorithm that optimizes personalization performance rather than the global model.

## Interpreting FedAvg as a Meta Learning Algorithm

The figure below shows MAML applied in the FL setting (left), the Reptile algorithm (middle), and FedAvg, the training algorithm of FL (right). Let L be the loss function. In each round of iteration, MAML trains by randomly sampling a batch of tasks T. For each task T there is an inner loop, and the outer loop aggregates the gradient updates obtained from each task. The FL algorithm randomly samples several clients T. For each T and its weight, several rounds of local optimization are performed, and the updated gradients are aggregated into a new global model. If we simplify the setting and assume all clients have the same amount of data, all weights become equal, and Reptile and FedAvg are then effectively the same algorithm.

![](http://cdn.ereebay.me/blog/fl-maml/fl-maml-1.png)

Assume the weights in FedAvg are equal, denoted wi. Consider T clients, and let the model parameters be $\theta$. For each client i, the loss function is $L_{i}(\theta)$; let $g_{j}^{i}$ denote the gradient computed during the $j^{t h}$ local training step.

The gradient update of FedSGD is:

$$
g_{F e d S G D}=\frac{-\beta}{T} \sum_{i=1}^{T} \frac{\partial L_{i}(\theta)}{\partial \theta}=\frac{1}{T} \sum_{i=1}^{T} g_{1}^{i}
$$

Now let us express FOMAML in the same terms. Assume the client learning rate is $\beta$; after K steps, the personalized model of client i obtains the updated parameters $\theta_{K}^{i}=U_{K}^{i}(\theta)=\theta-\beta \sum_{j=1}^{K} g_{j}^{i}=\theta-\beta \sum_{j=1}^{K} \frac{\partial L_{i}\left(\theta_{j}\right)}{\partial \theta}$

Differentiating gives:

$$
\frac{\partial U_{K}^{i}(\theta)}{\partial \theta}=I-\beta \frac{\partial \sum_{j=1}^{K} g_{j}^{i}}{\partial \theta}=I-\beta \sum_{j=1}^{K} \frac{\partial^{2} L_{i}\left(\theta_{j}\right)}{\partial \theta^{2}}
$$

After K gradient updates, the whole model is updated:

$$
g_{M A M L}=\frac{\partial L_{M A M L}}{\partial \theta}=\frac{1}{T} \sum_{i=1}^{T} \frac{\partial L_{i}\left(U_{K}^{i}(\theta)\right)}{\partial \theta}=\frac{1}{T} \sum_{i=1}^{T} L_{i}^{\prime}\left(U_{K}^{i}(\theta)\right)\left(I-\beta \sum_{j=1}^{K} \frac{\partial^{2} L_{i}\left(\theta_{j}\right)}{\partial \theta^{2}}\right)
$$

To avoid the computational cost of second-order derivatives, FOMAML was introduced: after K gradient updates, it directly uses the (K+1)-th gradient as the local update.

$$
g_{F O M A M L}(K)=\frac{1}{T} \sum_{i=1}^{T} L_{i}^{\prime}\left(U_{K}^{i}(\theta)\right) I=\frac{1}{T} \sum_{i=1}^{T} L_{i}^{\prime}\left(\theta_{K}^{i}\right)=\frac{1}{T} \sum_{i=1}^{T} g_{K+1}^{i}
$$

From the formulas above, it is easy to see that the FedAvg update — the average of all clients' updates — is essentially a linear combination of the two ideas above.

$$
g_{F e d A v g}=\frac{1}{T} \sum_{i=1}^{T} \sum_{j=1}^{K} g_{j}^{i}=\frac{1}{T} \sum_{i=1}^{T} g_{1}^{i}+\sum_{j=1}^{K-1} \frac{1}{T} \sum_{i=1}^{T} g_{j+1}^{i}=g_{F e d S G D}+\sum_{j=1}^{K-1} g_{F O M A M L}(j)
$$

## Personalized FedAvg

![](http://cdn.ereebay.me/blog/fl-maml/fl-maml-2.png)

As shown in the figure above, FedAvg E in Algorithm 1 trains for E local epochs, weighting the gradient updates by the local data volume. Then, in the FL setting, Reptile (K) trains for K local steps, ignoring the local data volume.

Generally speaking, in terms of the number of communication rounds, FedAvg with several local epochs converges quickly within just a few rounds of communication. Due to the complexity of production environments, this metric is used to measure the convergence speed of FL algorithms. This paper finds that using momentum SGD as the server optimizer already benefits the personalized model, whereas the initial model remains relatively unstable. Previous approaches reduced the number of local training epochs or the learning rate.

This paper proposes using Reptile (K) for fine-tuning and then Adam as the server optimizer, which improves the initial model while also stabilizing the personalized model.

To be continued
