---
title: Federated Adversarial Domain Adaptation 论文笔记【未完】
abbrlink: 62459
date: 2019-11-24 14:04:02
tags:
categories: notes
---

# Federated Adversarial Domain Adaptation

目前联邦学习存在知识域迁移的问题，而导致无法具有较好的泛化能力。当源节点的标记数据和目标节点的未标记数据不同时候，就会出现域迁移的情况。

这paper主要提出了一种方式来解决联邦学习中知识域使用的方法，为了能够将不同节点所学习的知识能够和目标节点的数据分布所对齐。

该论文的方法主要将对抗适应技术应用到了联邦学习中。此外，设计了一种动态注意力机制，并且利用特征分解来增强知识迁移。

<!-- more -->
## Introduction

传统的联邦学习存在一个问题：由于每个节点上的数据是通过nonidd的行为所收集而来的，因此会产生一个domain shift 的问题。例如：一个设备拍到的图片很多都是室内场景，一个拍到的很多是室外场景。本文主要提出的一个方法是，**在不需要额外用户的监督情况下，通过对去中心化的节点中的数据进行知识迁移到一个新的不同的数据域的节点。** 该方法也被称为，Unsupervised Federated Domain Adaptation.

目前有许多非监督域适应的方法，但是因为联邦学习的背景导致了一下的问题：

1. 数据的存放是locally，而且无法分享。
2. 模型的参数是在不同节点上独立训练，并且以不同的速度进行收敛，对于global模型的贡献取决于这两者域的相关性。
3. 由于通过源节点学习出来的知识是高度集成的，难以解构，可能会引起negative transfer。

本文主要提出的方法名称为Federated Adversarial Domain Adaptation(FADA)，利用对抗技术在联邦学习系统中解决域迁移的问题。

方法的主要步骤分为：

1. 首先从理论角度对联邦域适应问题进行分析，并且提供一个泛化的，通用的界限。
2. 收到理论效果的启发，提出一种有效的自适应算法，该算法基于对抗性适应和应用于联邦学习环境的表征解构技术。
3. 设计一个动态注意力模型来应对联邦学习系统中不断变化的收敛速率。

![png1](http://cdn.ereebay.me/FADAfada-1.png)

## Realated Work

**Unsupervised Domain Adaptation**: UDA的目的是将知识从已经标记的数据域迁移到一个未标记的数据域。

**Federated Learning**:不多介绍了

**Feature Disentanglement**:特征解构，神经网络通过复杂的隐藏层提取出来的特征，通常是高度集成的。因此通过学习解构的特征可以帮助一出一些不相关，或者是特定域的知识，然后对需要的域知识进行建模。

## Generalization Bound for FDA

1. 首先回顾一下以单数据源为背景的自适应问题的理论误差范围的定义
2. 然后描述在无监督的联邦域自适应情况下的误差范围的定义

## Federated Adversarial Domain Adaptation

![](http://cdn.ereebay.me/flpaper-2.png)

根据上一章节的理论不难看出权重$\alpha$和距离的重要性。本文的方法是通过提出一个动态的注意力机制进行权重的学习，通过联邦对抗对齐机制来缩小源域和目标域的距离。此外，还采用了特征解构的方法提取和域无关的特征来加强知识的迁移。

**动态注意力机制**：

该机制的原理主要是去提高那些贡献度高的节点权重，降低贡献度低的节点权重，那么如何判断节点的贡献程度呢？本文采用了gap statistics方法评估目标特征能够多好的被clustered。

$$
I=\sum_{r=1}^{k} \frac{1}{2 n_{r}} \sum_{i, j \in C_{r}}\left\|f_{i}^{t}-f_{j}^{t}\right\|_{2}
$$

假设有$C_{1}, C_{2}, \ldots, C_{k}$这么多聚集，$C_r$表示第r个聚集中，对象的索引，而且$n_{r}=\left|C_{r}\right|$。

直觉上来说，一个更小的gap statistics值说明了特征分布拥有更小的类内方差。通过计算两次迭代的gap statistics的差值来衡量每个源节点贡献的程度。

$$
I_{i}^{g a i n}=I_{i}^{p-1}-I_{i}^{p}
$$

它表示的是在目标域采用源域的梯度更新后，clusters可以提升多少。对于梯度的更新采用，来决定每个梯度贡献多少

$$
\text { Softmax }\left(I_{1}^{\text {gain }}, I_{2}^{\text {gain }}, \ldots, I_{N}^{\text {gain}}\right)
$$



**联邦对抗对齐**：

机器学习的模型会因为域距离，使得模型效果大大减弱。为解决这个问题，本文方法在联邦学习的环境中，改善了传统的对抗训练方法。在联邦学习的背景中，本文将对抗对齐优化过程分为两个步骤，1. 对于每个域，训练一个local的特征提取器，$G_i$和$G_t$分别对应$D_i$和$D_t$。2.对于每个（$D_i, D_t$）源-目标域对，训练一个对抗域鉴别器DI，来采用对抗学习的方式将两个分布进行对齐。首先训练DI识别特征来自于哪个特征域，然后训练生成器也就是特征提取器（$G_i, G_t$）来迷惑DI。注意：D只能获得$G_i$和$G_t$的输出。

在给定第i个源数据域$\mathbf{X}^{S_{i}}$，目标域$\mathbf{X}^T$，$D I_{i}$的优化目标为：

$$
\underset{\Theta^{D I_{i}}{L}}{L_{a d v_{D I}}\left(\mathbf{X}^{S_{i}}, \mathbf{X}^{T}, G_{i}, G_{t}\right)=-\mathbb{E}_{\mathbf{x}^{s_{i} \sim \mathbf{X}^{s_{i}}}}\left[\log D I_{i}\left(G_{i}\left(\mathbf{x}^{s_{i}}\right)\right)\right]-\mathbb{E}_{\mathbf{x}^{t} \sim \mathbf{x}^{t}}\left[\log \left(1-D I_{i}\left(G_{t}\left(\mathbf{x}^{t}\right)\right)\right)\right]}
$$

然后维持D不变，更新G的目标函数：

$$
\underset{\Theta^{G}{G}_{i}, \Theta^{G_{t}}}{L}\left(\mathbf{X}^{S_{i}}, \mathbf{X}^{T}, D I_{i}\right)=-\mathbb{E}_{\mathbf{x}^{s_{i} \sim \mathbf{X}^{s_{i}}}\left[\log D I_{i}\left(G_{i}\left(\mathbf{x}^{s_{i}}\right)\right)\right]-\mathbb{E}_{\mathbf{x}^{t} \sim \mathbf{X}^{t}}\left[\log D I_{i}\left(G_{t}\left(\mathbf{x}^{t}\right)\right)\right]}
$$

**特征解构**：

本文在对抗学习框架之下同时还采用了对抗解构的方式提取和域不变的特征。本文认为可以将提取的特征分为域特定和域不变特征。正如第一张图所示，特征解构器D会将特征分为两种类别，首先训练一个K分类的分类器，和K分类的类别识别器来根据特征预测标签。

$$
\begin{array}{c}{L_{cross entropy}} \\ {\Theta^{G_i}, \Theta^{D_i}, \Theta^{C_i}, \Theta^{CI_i}}\end{array}=-\mathbb{E}_{\left(\mathbf{x}^{s_i}, \mathbf{y}^{s_i}\right) \sim \widehat{\mathcal{D}}_{s_i}} \sum_{k=1}^{K} \mathbb{1}\left[k=\mathbf{y}^{s_{i}}\right] \log \left(C_{i}\left(f_{d i}\right)\right)-\mathbb{E}_{\left(\mathbf{x}^{s} i, \mathbf{y}^{s} i\right) \sim \widehat{\mathcal{D}}_{s_{i}}} \sum_{k=1}^{K} \mathbb{1}\left[k=\mathbf{y}^{s_{i}}\right] \log \left(C I_{i}\left(f_{d s}\right)\right)
$$

然后固定类别分类器，通过产生域特定特征来仅训练特征结构器，来迷惑类别分类器。

$$
\underset{\Theta^{D} i, \Theta^{G_{i}}}{L_{e n t}}=-\frac{1}{N_{s_{i}}} \sum_{j=1}^{N_{s_{i}}} \log C I_{i}\left(f_{d s}^{j}\right)=-\frac{1}{N_{s_{i}}} \sum_{j=1}^{N_{s_{i}}} \log C I_{i}\left(D_{i}\left(G_{i}\left(\mathbf{x}^{s_{i}}\right)\right)\right)
$$

特征解构方法通过保留类不变特征，和去除类特定特征来进行知识的迁移，为了增强解构的效果，本文通过最小化两者的共同信息。

共同信息定义为：$I\left(f_{d i} ; f_{d s}\right)=\int_{\mathcal{P} \times \mathcal{Q}} \log \frac{d \mathbb{P}_{\mathcal{P} Q}}{d \mathbb{P}_{\mathcal{P}} \otimes \mathbb{P}_{\mathcal{Q}}} d \mathbb{P}_{\mathcal{P} \mathcal{Q}}$，其中$\mathbb{P}_{\mathcal{P} \mathcal{Q}}$是两种特征的联合概率分布，$\mathbb{P}_{\mathcal{P}}=\int_{\mathcal{P}} d \mathbb{P}_{\mathcal{P} \mathcal{Q}}, \mathbb{P}_{\mathcal{Q}}=\int_{\mathcal{Q}} d \mathbb{P}_{\mathcal{P} \mathcal{Q}}$分别是对应的边缘概率分布。

采用Mutual Information Neural Estimator（MINE）来评估互信息：

$$
T_{\theta}: \widehat{I(\mathcal{P} ; \mathcal{Q})}_{n}=\sup _{\theta \in \Theta} \mathbb{E}_{\mathbb{P}_{\mathcal{P} Q}^{(n)}}\left[T_{\theta}\right]-\log \left(\mathbb{E}_{\mathbb{P}_{P}^{(n)} \otimes \mathbb{P}_{Q}^{(n)}}\left[e^{T_{\theta}}\right]\right)
$$

实际情况下可以通过下式进行计算：

$$
I(\mathcal{P} ; \mathcal{Q})=\iint \mathbb{P}_{\mathcal{P} \mathcal{Q}}^{n}(p, q) T(p, q, \theta)-\log \left(\iint \mathbb{P}_{\mathcal{P}}^{n}(p) \mathbb{P}_{\mathcal{Q}}^{n}(q) e^{T(p, q, \theta)}\right)
$$

为了避免计算积分，采用门特卡罗法还避免计算积分：

$$
I(\mathcal{P}, \mathcal{Q})=\frac{1}{n} \sum_{i=1}^{n} T(p, q, \theta)-\log \left(\frac{1}{n} \sum_{i=1}^{n} e^{T\left(p, q^{\prime}, \theta\right)}\right)
$$

其中（p，q）采样自于联合分布，$q^{\prime}$采样与边缘分布，$T(p, q, \theta)$是由$\theta$决定的神经网络，来评估P和Q分布的互信息。

域无关特征和域特定特征会输入到重建器中得到重组的特征，loss函数采用l2损失，来重建原来特征，为了保持表示的可组成型。L2约束和互信息的损失约束可以通过调整超参数来平衡。


To be continued