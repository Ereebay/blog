---
title: "Watermelon Book Reading Notes 1. Introduction"
date: 2020-06-24T18:15:38+08:00
draft: false
math: true
categories: [notes]
tags: [watermelon book]
---
# Introduction

## Preamble

Machine learning: suppose we use P to evaluate the performance of a computer program on a certain class of tasks T; if a program improves its performance on the tasks in T through experience E, then we say that, with respect to T and P, the program has learned from E.

<!-- more -->
## Basic Terminology

Data-related:

A dataset contains a series of records, and each record is an instance or a sample. That which reflects certain properties is an attribute or a feature. The value an attribute takes is called the attribute value, and the space spanned by the attributes is called the attribute space, the sample space, or the input space. Every point in this space is called a feature vector.

Training-related:

The process of learning a model from data is called learning or training. The data used is called training data, each sample is a training sample, and the whole collection is called the training set. The learned model corresponds to some underlying regularity in the data, which is called a hypothesis. The regularity itself is called the ground truth. The model can also be called a learner.

After training, predictions need to be made. Information about the outcome of an instance is called a label, and an instance with a label is an example. The space spanned by the labels is the label space or the output space.

Classification: the predicted values are discrete. Regression: the predicted values are continuous. Binary classification: the outputs are the positive class and the negative class.

The stage of making predictions with the learned model is called testing; the samples being predicted are called testing samples, and likewise there is a testing set

Supervised learning: training with labeled data, such as classification and regression.

Unsupervised learning: training with unlabeled data, such as clustering, which divides training samples into clusters.

Generalization: the ability of a model to work well on new samples

Independent and identically distributed (iid): the samples are all drawn independently from the same distribution.

## Hypothesis Space

Induction: from the specific to the general. (generalization)

Deduction: from the general to the specific. (specialization)

Inductive learning: in the broad sense, learning from samples; in the narrow sense, acquiring concepts from data, which is called concept learning.

Version space: the set of hypotheses consistent with the training set

## Inductive Bias

Inductive bias: the preference of a machine learning algorithm for a certain type of hypothesis during the learning process; it corresponds to the assumption the learning algorithm itself makes about "what kind of model is better"
