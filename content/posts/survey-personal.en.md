---
title: "A Survey on Personalization Capabilities in Federated Learning"
date: 2020-04-29T12:37:55+08:00
draft: false
categories: [notes]
tags: [notes]
---

# A Survey on Personalization in Federated Learning

## Abstract

The goal of federated learning is to obtain a shared global model that all nodes can use. However, due to Non-IID data distributions, local models trained solely on local data often outperform the global model, which makes these nodes reluctant to participate in the federation process. This post introduces techniques currently used to personalize the global model so as to improve its performance on individual nodes.

<!-- more -->
## Introduction

Federated learning is a distributed machine learning algorithm designed for Non-IID data. It enables multiple nodes to collaboratively train a shared global model without sharing their own data. At the beginning of each training round, the central server sends the current global model to every node, and each node trains the model on its local data. The central server then collects the model updates from all nodes and applies them to the global model.

The main reason for a node to participate in federated learning is to obtain a better model. When a node does not have enough data to train a good local model on its own, it can obtain a decent model through federated learning. But for nodes with abundant data, the benefit of federated learning remains in question. Yu, in Salvaging Federated Learning by Local Adaptation, noted that for many tasks some nodes gain no benefit because the globally shared model underperforms a model trained on their local data. Hanzely, in Federated Learning of a Mixture of Global and Local Models, questioned whether the global model deviates too far from the classical use cases of users.

The purpose of this post is to survey recent techniques for building personalized models on local nodes, where the personalized models are expected to outperform either the shared global model or the model trained locally by the node.

## Need for Personalization

Wu, in Personalized federated learning for intelligent iot applications, described the three major challenges federated learning systems face with respect to personalization:

1. Heterogeneity of devices in terms of storage, computation, and communication capabilities.
2. Data heterogeneity caused by Non-IID distributions.
3. Model heterogeneity arising from the need for different models due to different user environments.

To address the challenges brought by data heterogeneity and Non-IID data, personalizing the global model has become increasingly important. Most personalization techniques consist of two steps: 1. learn a global model collaboratively across the nodes; 2. personalize the global model using each node's private data. Jiang argued that optimizing solely for global-model accuracy hardly produces models that personalize well, and proposed that, for personalization in federated learning to be useful in practice, the following three objectives must be solved simultaneously rather than independently:

1. Build improved personalized models that benefit the majority of nodes
2. Build a more accurate global model, which enables better personalization for nodes with little data
3. Achieve fast model convergence within few training rounds

In addition, only a fraction of the samples in a node's local data may be relevant to the specific task, while the remaining samples are irrelevant ones that hurt model training. Tuor proposed a method that uses a relevance model built on a small benchmark to distinguish relevant from irrelevant samples, and then uses only the relevant samples during federated learning.

## Techniques

### Adding User Context

Before introducing methods for personalizing the global model for each node, it must be pointed out that a shared global model can also make highly personalized predictions, provided that the node's context and personal information are properly featurized and integrated into the dataset. However, most public datasets do not contain contextual features, and developing techniques that can effectively incorporate context remains an important open problem—one with great potential to improve federated learning models. At the same time, whether such contextual featurization can be done without adversely affecting privacy remains to be studied. As an intermediate approach between a single global model and purely local models, Masour, in Three approaches for personalization with applications to federated learning, suggested clustering users—grouping similar clients together—and training a separate model for each group.

### Transfer Learning

Transfer learning allows deep learning models to leverage the capabilities acquired while solving problem A to solve another related problem. Schneider and Vlachos, in Mass personalization of deep learning, discussed using transfer learning to achieve model personalization in non-federated settings. Transfer learning has likewise been applied in federated settings. Wang, in Federated evalua- tion of on-device personalization, proposed re-learning some or all parameters of the trained global model on local data; the previously cited work provides the theoretical justification. By initializing local training with the parameters of the trained global model, transfer learning exploits the knowledge extracted by the global model instead of learning from scratch. Meanwhile, to avoid catastrophic forgetting, care must be taken not to retrain/finetune on local data for too long. One variant of this technique freezes the base layers of the global model and trains only the top layers on local data (similar to freezing the feature layers and finetuning the fully connected layers).

### Multi-task Learning

In multi-task learning, a model exploits the commonalities and differences across tasks through joint learning, so that multiple related tasks can be solved simultaneously. Smith, in Federated multi-task learning, showed that multi-task learning is a natural choice for constructing personalized federated models and developed the MOCHA algorithm. A drawback of using multi-task learning in federated settings, however, is that, because it produces one model per task, all nodes must participate in every round of training.

### Meta Learning

Meta-learning involves training across multiple learning tasks to produce models that adapt quickly—models that can fit and solve a new task with only a few training samples. Finn proposed the Model-Agnostic Meta-Learning algorithm (MAML), which is compatible with any model trained by gradient descent. MAML builds internal representations that work across multiple tasks, so for a new task, fine-tuning the top layers can produce good results.

Jiang pointed out that the federated learning process can be viewed as meta-training, while the personalization process can be viewed as meta-testing; the FedAvg algorithm is then very similar to Reptile. The author also observed that careful fine-tuning can yield accurate global models that are easy to personalize, whereas optimizing purely for global-model accuracy sacrifices the model's subsequent personalization capability. Other personalization approaches in federated learning treat the construction of the global model and personalization as two separate processes; Jiang proposed an improved FedAvg algorithm that achieves both a better global model and better personalized models.

Fallah, in Personalized federated learning: A meta-learning approach, formulated a new version of the standard federated learning problem that incorporates MAML, seeking a global model that performs well on every node after being updated with that node's own loss function; they further proposed Per-FedAvg to solve this problem. Khodak, in Adaptive gradient-based meta-learning methods, proposed ARUBA and demonstrated improved performance by applying it to FedAvg. Chen, in Federated meta-learning for recommendation, proposed a federated meta-learning framework for building personalized recommendation models, in which both the algorithm and the model are parameterized and need to be optimized.

### Knowledge Distillation

Caruana showed that the knowledge of an ensemble of models can be compressed into a single model that is easy to deploy. Knowledge distillation develops this idea further, extracting the knowledge of a large teacher network into a smaller student network by having the student imitate the teacher. Overfitting poses a major obstacle during personalization, especially for nodes with small local datasets. Yu proposed that treating the global model as the teacher and the node's personalized model as the student can alleviate overfitting. Li proposed FedMD, a federated learning framework based on knowledge distillation and transfer learning that allows nodes to independently design their own networks while leveraging both their private data and a global public dataset.

### Base+Personalization Layers

In the conventional federated learning setting, data is distributed across the multiple devices participating in training. To mitigate the adverse effects of such data heterogeneity, Arivazhagan et al., in Federated learning with personalization layers, proposed the FedPer architecture, in which the base layers are trained with FedAvg while the top layers (the personalization layers) are trained on the local node with a variant of gradient descent. The transfer-learning approach first trains all layers on global data and then retrains some or all layers on the local node's data, whereas FedPer trains the base layers on global data and the personalization layers on local data, separately.

### Mixtrue of Global and Local Models

Hanzely, in Federated learning of a mixture of global and local models, proposed a new problem formulation that seeks a balance between the global model and local models. Instead of learning a single global model, each node learns a mixture of the global model and its own local model. To solve this problem, the authors proposed LLGD, Loopless Local Gradient Descent (LLGD), a variant of gradient descent. Unlike full averaging, this method only takes averaged steps, because full averaging can be too aggressive.

## Discussion

In federated learning, when a node's local dataset is small and IID (isn't this similar to the meta-learning scenario?), the global model usually surpasses local models, and most nodes benefit from the federation process. However, when nodes possess sufficiently large private datasets with Non-IID distributions, local models usually outperform the global model, and nodes are generally reluctant to participate in federated learning. **An open theoretical question is: how to determine when the global model will outperform the model on a single node**

This post has mainly summarized several techniques for optimizing the personalization of the global model. With a few exceptions, most prior work has focused on measuring the performance of global models on aggregated data rather than on individual nodes. But if the global model is to be personalized before use, aggregate global performance becomes meaningless.

Personalized models usually outperform both the global model and purely local models on individual nodes. In some cases, however, a personalized model cannot reach the same capability as a purely local model, especially under differential privacy and similar conditions.
