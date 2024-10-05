---
title: 【西瓜书】阅读笔记 1. 绪论
tags: 西瓜书
categories: notes
mathjax: true
abbrlink: 32073
date: 2020-06-24 18:15:38
---
# 绪论

## 引言

机器学习：假设用P来评估计算机程序在某任务类T上的性能，若一个程序通过经验E在T中任务上获得了性能改善，则我们就说关于T和P，该程序对E进行了学习。

<!-- more -->
## 基本术语

数据相关：

数据集（dataset）包含了一系列的记录，每条记录就是一个示例（instance）或者是样本（sample）。反应某些性质的是属性（attribute）或者是特征（feature）。属性取值称为属性值（attribute value），其张成空间称为属性空间（attribute space）或者样本空间（sample space）或者输入空间。空间中每一个点称为特征向量（feature vector）。

训练相关：

从数据学习到模型的过程称为学习（learning）或训练（training）。使用的数据称为训练数据（training data），每个样本为训练样本（training sample），整个集合称为训练集（training set）。学的模型对应了关于数据的某种潜在规律称为假设（hypothesis）。规律本身称为真相（ground truth）。模型也可以称为学习器（learner）。

训练之后，需要进行预测（prediction），关于示例结果的信息称为标签（label），拥有标签的示例是样例（example）。标签张成的空间为标记空间或者输出空间。

分类任务（classification）：预测值为离散值。回归任务（regression）：预测值为连续值。二分类任务（binary classification）：输出为正类和负类。

学的模型后进行预测的阶段为测试（testing），被预测的样本称为测试样本（testing sample），同理还有测试集（testing set）

有监督学习（supervised learning）：用拥有标记的数据训练，如分类和回归。

无监督学习（unsupervised learning）：无标签的数据训练，如聚类（clustering），将训练样本分成若干个簇（cluster）。

泛化（generalization）：模型适用于新样本的能力

独立同分布（iid）：样本都从同一个分布上独立采样获得。

## 假设空间

归纳（induction）：从特殊到一般。（generalization）

演绎（deduction）：从一般到特殊。（specialization）

归纳学习（inductive learning）：广义：从样本中学习。狭义：从数据中获得概念（concept），称为概念学习。

版本空间（version space）：与训练集一致的假设集合

## 归纳偏好

归纳偏好（inductive bias）：机器学习算法在学习过程中对某种类型假设的偏好，对应了学习算法本身所做出的关于“什么样的模型更好的”假设
