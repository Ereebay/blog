---
title: MemoryGAN
mathjax: true
categories: notes
abbrlink: 64315
date: 2019-04-22 14:05:02
tags:
---
# Memory GAN 阅读笔记

## 简介

本文主要解决了训练非监督GAN中的两个问题，

第一，由于生成对抗网络只使用连续潜在分布来表示多个类或者数据簇，因此他们通常没办法正确的处理潜在空间中不同类之间的结构不连续性质。(由于模式崩溃问题导致) 例如 GAN 吧建筑和猫 嵌入在同一个连续潜在分布中 因此GAN可能在两个类别的过渡区域中 生成真实图像。

其次， 生成对抗网络的鉴别器非常容易遗忘过去生成的样本，在对抗训练过程中会产生不稳定性。

作者认为这两个主要问题可以通过生成器鉴别器都能访问的可学习的记忆网络得到解决。生成器可以有效学习训练样本的表示，以理解数据的底层聚类分布，从而缓解结构的不连续问题。与此同时，鉴别器可以更好记忆先前生成的样本的集群，这可以减轻遗忘的问题。

本文提出了一种端到端的生成对抗网络模型 记忆GAN，该模型涉及一种无监督的并且和现有生成对抗网络模型继集成的存储网络。

<!-- more -->

Von Mises-Fisher (vMF) mixture model. 
记忆模块能够有效地缓解不稳定的问题。首先，为缓解结构不连续性问题，内存可以学习训练样本的表示，帮助生成器更好理解类和集簇分布。因此，我们可以将离散簇的建模与连续潜在空间上的数据属性的嵌入分开，这可以减轻不连续性问题。

其次 记忆网络能够通过学习记忆先前生成的样本数据簇来缓解遗忘问题，包括那种很稀有的样本。


## 结构

总体结构分为 记忆鉴别网络DMN和记忆条件生成网络（MCGN）

### 记忆鉴别网络

记忆鉴别网络分别由一个前馈网络$\mu$ 和一个记忆网络组成。

其中前馈网络 $\mu$ 是卷积神经网络，输入为$x \in \mathbb{R}^{D}$，输出是一个标准化后的向量 $q=\mu(x) \in \mathbb{R}^{M}$，其中 $\|q\|=1$。然后作为记忆模块的输入并输出判断结果。

记忆网络的公式为：$\mathcal{M}=(K, v, a, h)$，其中$K \in \mathbb{R}^{N \times M}$ 是内存key矩阵，$N$是内存尺寸而$M$是维度。$v \in\{0,1\}^{N}$ 是内存矩阵。从概念上将，每个关键向量存储通过vMF混合模型学习的聚类中心表示，并且其对应的关键值是真假的判断结果。$a \in \mathbb{R}^{N}$ 是表示存在每个内存块中的物体的时长。而$h \in \mathbb{R}^{N}$表示的是slot 直方图，其中每个$h_{i}$表示属于第i个内存槽中的有效数据量。

### 相关机制：

life-long memory network：可以自由增长内存空间，训练过程中无需重置。

k-nearest neighbor indexing for efficient memory lookup:
k近邻索引用于查询

least recently used (LRU) scheme for memory update.
LRU机制用于内存更新

### 创新：

- 概率推倒：可以计算数据极大似然，内存索引的范畴分布的先验分布后验分布。
- memory通过使用增量EM算法最大化似然来学习query的近似分布。
- 通过GANloss优化memory而不是memoryloss优化
- 通过跟踪slot histogram 来决定每个样本的贡献程度

### 鉴别器输出

对于每个输入$x$， 需要先决定哪个memory slot 来计算discriminative probability

$c \in\{1,2, \ldots, N\}$ 表示 memory slot index。

采用Von Mises-Fisher(vMF)混合模型表示memory index 的后验分布。

$$p(c=i | x)=\frac{p(x | c=i) p(c=i)}{\sum_{j=1}^{N} p(x | c=j) p(c=j)}=\frac{\exp \left(\kappa K_{i}^{T} \mu(x)\right) p(c=i)}{\sum_{j=1}^{N} \exp \left(\kappa K_{j}^{T} \mu(x)\right) p(c=j)}$$

$p(x | c=i)=C(\kappa) \exp \left(\kappa K_{i}^{T} \mu(x)\right)$中 $\kappa=1$ 为常量注意力参数。

vMF实际上等效于在单位球面上定义的正确归一化的高斯分布

关于memory index的范畴分布 $p(c)$ 通过归一化 slot histogram $p(c=i)=\frac{h_{i}+\beta}{\sum_{j=1}^{N}\left(h_{j}+\beta\right)}$，其中$\beta\left(=10^{-8}\right)$是一个平滑常量用于数值稳定性。通过使用$p(y=1 | c=i, x)=v_{i}$，我们在c上求边缘化联合概率$p(y=1, c | x)$ 得到$p(y=1 | x)$：

$$
p(y=1 | x)=\sum_{i=1}^{N} p(y=1 | c=i, x) p(c=i | x)=\sum_{i=1}^{N} v_{i} p(c=i | x)=\mathbb{E}_{i \sim p(c | x)}\left[v_{i}\right]
$$

然而对于每个样本x，对于整个大小为N的memory是不可扩展的。

采用最大后验概率来考虑top-k slots $S=\left\{s_{1}, 
\dots, s_{k}\right\}$

$$
S=\underset{c_{1}, \ldots, c_{k}}{\operatorname{argmax}} p(c | x)=\underset{c_{1}, \ldots, c_{k}}{\operatorname{argmax}} p(x | c) p(c)=\underset{c_{1}, \ldots, c_{k}}{\operatorname{argmax}} \exp \left(\kappa K_{c}^{T} \mu(x)\right)\left(h_{c}+\beta\right)
$$

其中$p(x | c)$是vMF似然，$p(c)$是memory index的先验分布。

在这我们省略了vMF似然和先验坟墓的归一化，因为他在都是常数。一旦我们获得了S，就可以求得

$$
p(y | x) \approx \frac{\sum_{i \in S} v_{i} p(x | c=i) p(c=i)}{\sum_{j \in S} p(x | c=j) p(c=j)}
$$

### 内存更新机制

memory keys and values 在训练过程中会进行更新。更新机制包括传统的内存更新机制和增量EM算法。

设样本为$x$标签$y$真为1假为0。对于每一个x，首先寻找k-nearest slots $S_{y}$，但是采用条件后验$p\left(c | x, v_{c}=y\right)$。 这是为了能够在接下去的EM算法过程中，只考虑与y属于同一个类别的slots。

后面我们就根据$S_y$中是否包含正确的标签来根据不同方式更新memory。

#### 没有正确label：

通过$n_{a}=\operatorname{argmax}_{i \in\{1, \ldots, N\}} a_{i}$寻找最老的memory slot，并将对x的信息复制到对应位置。$K_{n_{a}} \leftarrow q=\mu(x)$，$v_{n_{a}} \leftarrow y$，$a_{n_{a}} \leftarrow 0$，$h_{n_{a}} \leftarrow \frac{1}{N} \sum_{i=1}^{N} h_{i}$。

#### 有正确label：

通过以下用于T次迭代的自定义的增量EM算法，部分包括新样本的信息来更新memory key。

在推断过程中，通过将前一时刻的keys$\hat{K}_{i}^{t-1}$和$\hat{h}_{i}^{t-1}$应用到公式1，计算后验分布$\gamma_{i}^{t}=p\left(c_{i} | x\right)$，其中$i \in S_{y}$。在最大化过程中，将进行如下更新：

$$
\hat{h}_{i}^{t} \leftarrow \hat{h}_{i}^{t-1}+\gamma^{t}-\gamma^{t-1}, \quad \hat{K}_{i}^{t} \leftarrow \hat{K}_{i}^{t-1}+\frac{\gamma^{t}-\gamma^{t-1}}{\hat{h}_{i}^{t}}\left(q_{i}-\hat{K}_{i}^{t}\right)
$$

其中，$t \in 1, \ldots, T, \gamma^{0}=0, \hat{K}_{i}^{1}=K_{i}, \hat{h}_{i}^{1}=\alpha h_{i}$，$\alpha=0.5$

经过T次迭代，通过$K_{i} \leftarrow \hat{K}_{i}^{t}$和$h_{i} \leftarrow \hat{h}_{i}^{t}$更新$S_y$的slots。

衰减率$\alpha$控制它以指数方式减少就查询对混合分量的平均方向的slot位置的贡献的程度。

$\alpha$对于性能至关重要，因为用于更新keys的旧序列不适合当前的混合粉不，因为前馈网络本身也在进行更新。

最终，值得注意的是，这种内存更新机制和对抗训练算法是正交的，因为当鉴别器更新的时候，是内存的更新是独立执行的。此外，添加内存模块不会影响模型在测试的速度，因为memory只在训练的时候更新

## 记忆条件生成网络

记忆条件生成网络基于InfoGAN的生成器。但是区别在于他不仅以随机噪声为条件，同样对memory信息为条件。

换言之，生成器不仅从噪声分布中随机采样枣红色呢过，同时还会从$P\left(c=i | v_{c}=1\right)=\frac{h_{i} v_{i}}{\sum_{j}^{N} h_{j} v_{j}}$采样memory index $i$。上面的公式表示的是存储真实数据的单元i出现的频率。最终的输入是$[K_i, z]$，$K_i$是memory index i的key vector

与其他CGAN的区别是，MCGN不需要额外的标注或者是额外的encoder。相反，MCGN可以充分利用DMN通过非监督形式学习到的memory信息。DMN仅采用序列中每个样本和其标签来学习vMF混合memory


整个memorygan的训练过程是

for 训练迭代次数 do
    从训练样本中采样一个minibatch的样本
    从噪声分布和memory index 随机采样一个minibatch
    更新鉴别器的loss
    寻找minibatch中每个数据的Sy
    对Sy中的每个key h gama 进行初始化
    for EM迭代次数 do
        估计每个s的$\gamma_s$
        更新$h_s$
        更新$K_s$
    更新 vMF混合模型， $h_{s_{y}} \leftarrow \hat{h}_{s_{y}}^{T}, K_{s_{y}} \leftarrow \hat{K}_{s_{y}}^{T}$ for $s_{y} \in S_{y}$
    从噪声分布和memory index 随机采样一个minibatch
    更新生成器loss

## objective function

memoryGAN的目的是基于InfoGAN的目的，是为了最大化潜在变量和观察内容之间的共同信息。（这个可以查阅infoGAN）

在$K_i$和$G(z, K_i)$之间增加一个共同信息loss，来保证采样的memory 信息和生成样本的结构化信息的连续性：

$$
I\left(K_{i} ; G\left(z, K_{i}\right)\right) \geq H\left(K_{i}\right)-\hat{I}-\log C(\kappa)
$$

其中，$\hat{I}$ 表示负的余弦相似度的期望 $\hat{I}=-E_{x \sim G\left(z, K_{i}\right)}\left[\kappa K_{i}^{T} \mu(x)\right]$

To be continued