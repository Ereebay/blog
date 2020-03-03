---
title: Improving Federated Learning Personalization via MAML 论文笔记 【未完】
abbrlink: 6350
date: 2019-12-02 15:07:30
tags:
categories: notes
---

# Improving Federated Learning Personalization via MAML

## Abstract

1. FL算法与MAML具有很多相似性，可以用元学习算法来对其进行解释
2. 微调可以使得gloabl 模型具有更强的准确率，同时更容易做定制化处理
3. 通过标准的中心化数据库训练出来的模型相比Fedavg训练的更难进行定制化处理
<!-- more -->
## Introduction

1. 指出了FL与MAML算法的联系，并用MAML算法对FL算法进行解释
2. 对FedAvg进行改进，采用两阶段的训练和fine-tune进行优化
3. 发现FedAvg其实本质是一种metalearning算法，用于优化个性化定制的效果，而不是全局模型的优化。

## Interpreting FedAvg as a Meta Learning Algorithm

下图展现了在FL中应用MAML算法（左侧），Reptile算法（中间）和FL的训练算法FedAvg（右侧）。设L为损失函数，在每一轮的迭代中，MAML会通过随机采样一个batch的任务T来进行训练。对于每个任务T，会有一个内循环，然后在外循环中聚集每个任务所获得的的梯度更新。对于FL算法会随机采样数个client T。对于每个T和其权重，会在local数据上进行数轮的迭代优化，然后将更新的梯度聚集形成一个新的global model。如果我们简化设置，并认为所有的client拥有相同的数据，那么所有的权重就会一样，这个时候reptile和fedavg其实就是同一种算法。

![](http://cdn.ereebay.me/blog/fl-maml/fl-maml-1.png)

假设在FedAvg中的权重相同为wi。考虑有T个clients，并设置每个相关模型参数为$\theta$。对于每个cilent i，其损失函数为$L_{i}(\theta)$,记$g_{j}^{i}$为第$j^{t h}$local训练过程所计算得到的梯度。

FedSGD的梯度更新函数为：

$$
g_{F e d S G D}=\frac{-\beta}{T} \sum_{i=1}^{T} \frac{\partial L_{i}(\theta)}{\partial \theta}=\frac{1}{T} \sum_{i=1}^{T} g_{1}^{i}
$$

将设我们将FOMAML用相同的术语来表示。假设Client 学习率为$\beta$, 每个client的个性化模型经过K步后所获得的梯度更新为$\theta_{K}^{i}=U_{K}^{i}(\theta)=\theta-\beta \sum_{j=1}^{K} g_{j}^{i}=\theta-\beta \sum_{j=1}^{K} \frac{\partial L_{i}\left(\theta_{j}\right)}{\partial \theta}$

求微分可得到：

$$
\frac{\partial U_{K}^{i}(\theta)}{\partial \theta}=I-\beta \frac{\partial \sum_{j=1}^{K} g_{j}^{i}}{\partial \theta}=I-\beta \sum_{j=1}^{K} \frac{\partial^{2} L_{i}\left(\theta_{j}\right)}{\partial \theta^{2}}
$$

在进行K次梯度更新后，对整个模型进行更新：

$$
g_{M A M L}=\frac{\partial L_{M A M L}}{\partial \theta}=\frac{1}{T} \sum_{i=1}^{T} \frac{\partial L_{i}\left(U_{K}^{i}(\theta)\right)}{\partial \theta}=\frac{1}{T} \sum_{i=1}^{T} L_{i}^{\prime}\left(U_{K}^{i}(\theta)\right)\left(I-\beta \sum_{j=1}^{K} \frac{\partial^{2} L_{i}\left(\theta_{j}\right)}{\partial \theta^{2}}\right)
$$

为了避免二次求导带来的计算量问题，FOMAML应运而生，通过K次的梯度更新后，直接采用第K+1次的梯度更新作为local update。

$$
g_{F O M A M L}(K)=\frac{1}{T} \sum_{i=1}^{T} L_{i}^{\prime}\left(U_{K}^{i}(\theta)\right) I=\frac{1}{T} \sum_{i=1}^{T} L_{i}^{\prime}\left(\theta_{K}^{i}\right)=\frac{1}{T} \sum_{i=1}^{T} g_{K+1}^{i}
$$

通过上面的公式，不难看出，其实FedAvg的更新，所有client的更新的平均，其实就是以上两种idea的线性组合。

$$
g_{F e d A v g}=\frac{1}{T} \sum_{i=1}^{T} \sum_{j=1}^{K} g_{j}^{i}=\frac{1}{T} \sum_{i=1}^{T} g_{1}^{i}+\sum_{j=1}^{K-1} \frac{1}{T} \sum_{i=1}^{T} g_{j+1}^{i}=g_{F e d S G D}+\sum_{j=1}^{K-1} g_{F O M A M L}(j)
$$

## Personalized FedAvg

![](http://cdn.ereebay.me/blog/fl-maml/fl-maml-2.png)

如上图所示，采用算法1中的FedAvg E训练E个local epoch，根据local数据量来对梯度更新进行权衡。然后在FL的环境下采用Retile（K）训练K个local steps，不考虑本地的数据量。

一般来说，就通信轮次的数量而言，FedAvg训练数个local epochs后，可以在数轮通信内就能快速收敛。由于生产环境的复杂性，这种测量方式被用于衡量FL算法的收敛速度。本文发现，采用momentum SGD的方法作为server优化器已经对于personalized model进行了优化，然而initial model相对不稳定。以前的方法是减少本地的训练轮次或者学习率。

本文提出采用Retile（K）的方法进行fintune，然后用Adam作为server优化器，来提升initialmodel的效果。同时可以稳定personalized model。

To be continued