---
title: "One-shot Learning with Memory-Augmented Neural Networks Paper Notes (Incomplete)"
date: 2019-03-22T08:47:11+08:00
draft: false
math: true
categories: [notes]
---
# One-shot Learning with Memory-Augmented Neural Networks Paper Notes

## Introduction

Traditional gradient-based deep learning methods require large amounts of data to learn. When confronted with new data, the model has to relearn new parameters and cannot quickly adapt to it.

This paper proposes a memory-augmented neural network that can rapidly assimilate new data and, after learning from only a few samples, use that data to make accurate predictions.

<!-- more -->

For the design of the MANN model, the authors had two requirements:

1. The stored information must be stable and addressed element-wise.
2. The number of parameters must not be tied to the size of the memory.

In the end, the model combines the following two advantages:

1. It can learn a general way of learning by extracting useful information from raw data via gradient descent.
2. It can rapidly learn never-before-seen information by relying on an additional memory module.

## Meta-Learning Method Design

Usually, parameters $\theta$ are chosen on some dataset $D$ to minimize a learning cost $\mathcal{L}$.

In meta-learning, however, what is minimized is the expected cost with respect to a distribution over datasets $p(D)$:

$$
\theta^{*}=\operatorname{argmin}_{\theta} E_{D \sim p(D)}[\mathcal{L}(D ; \theta)]
$$

To achieve this, the paper makes the following design choices:

1. First, the inputs are presented as a sequence, where each input carries the label of the previous input:

$$
\left(\mathbf{x}_{1}, \text { null }\right),\left(\mathbf{x}_{2}, y_{1}\right), \ldots,\left(\mathbf{x}_{T}, y_{T-1}\right)
$$

2. Across different datasets, the labels are shuffled (to prevent the network from gradually learning a fixed mapping between samples and labels). Instead, this forces the network to learn the trick of holding a data sample in memory until the correct label arrives, so that sample-label information is bound together and stored for later use.

![](/images/mann/mann-1.png)

Without relying on the actual content of the data and labels, the meta-learning model learns to bind the data distribution to the corresponding labels, and generalizes into a universal model that captures the data-label relationship for prediction.

## Memory-Augmented Model

### Neural Turing Machine

The composition of the Neural Turing Machine is largely similar to the MANN in this paper. An NTM consists of a controller, such as an LSTM or a feedforward network, which interacts with an external memory module through a set of read/write heads.

In this model, the controller likewise uses an LSTM or a feedforward network.

To be continued
