---
title: "MemoryGAN Paper Notes (Incomplete)"
date: 2019-04-22T14:05:02+08:00
draft: false
math: true
categories: [notes]
---
# Memory GAN Reading Notes

## Introduction

This paper mainly solves two problems in training unsupervised GANs.

First, since generative adversarial networks use only a continuous latent distribution to represent multiple classes or clusters of data, they usually cannot correctly handle the structurally discontinuous nature between different classes in the latent space. (Caused by the mode collapse problem.) For example, if a GAN embeds buildings and cats in the same continuous latent distribution, the GAN may generate realistic images in the transition region between the two classes.

Second, the discriminator of a generative adversarial network very easily forgets the samples generated in the past, which causes instability during adversarial training.

The authors believe that these two main problems can be solved by a learnable memory network accessible to both the generator and the discriminator. The generator can effectively learn representations of the training samples to understand the underlying clustering distribution of the data, thereby alleviating the structural discontinuity problem. Meanwhile, the discriminator can better memorize the clusters of previously generated samples, which can alleviate the forgetting problem.

This paper proposes an end-to-end generative adversarial network model, Memory GAN, which involves an unsupervised memory network integrated with existing generative adversarial network models.

<!-- more -->

Von Mises-Fisher (vMF) mixture model. 
The memory module can effectively alleviate the instability problem. First, to alleviate the structural discontinuity problem, the memory can learn representations of the training samples, helping the generator better understand the class and cluster distributions. Therefore, we can separate the modeling of discrete clusters from the embedding of data attributes on the continuous latent space, which can alleviate the discontinuity problem.

Second, the memory network can alleviate the forgetting problem by learning to memorize the clusters of previously generated sample data, including very rare samples.


## Architecture

The overall architecture is divided into the memory discriminative network (DMN) and the memory conditional generative network (MCGN).

### Memory Discriminative Network

The memory discriminative network consists of a feedforward network $\mu$ and a memory network.

The feedforward network $\mu$ is a convolutional neural network, whose input is $x \in \mathbb{R}^{D}$ and whose output is a normalized vector $q=\mu(x) \in \mathbb{R}^{M}$, where $\|q\|=1$. This is then used as the input of the memory module, which outputs the discriminative result.

The memory network is formulated as $\mathcal{M}=(K, v, a, h)$, where $K \in \mathbb{R}^{N \times M}$ is the memory key matrix, $N$ is the memory size and $M$ is the dimension. $v \in\{0,1\}^{N}$ is the memory value matrix. Conceptually, each key vector stores a representation of a clustering center learned through the vMF mixture model, and its corresponding key value is the real/fake judgment. $a \in \mathbb{R}^{N}$ denotes the age of the object stored in each memory block. And $h \in \mathbb{R}^{N}$ denotes the slot histogram, where each $h_{i}$ represents the amount of valid data belonging to the i-th memory slot.

### Related Mechanisms:

life-long memory network: the memory space can grow freely, and there is no need to reset it during training.

k-nearest neighbor indexing for efficient memory lookup:
k-nearest neighbor indexing is used for lookup.

least recently used (LRU) scheme for memory update.
The LRU scheme is used for memory update.

### Innovations:

- Probabilistic derivation: the data likelihood, as well as the prior and posterior distributions of the categorical distribution over memory indices, can be computed.
- The memory learns an approximate distribution of queries by maximizing the likelihood with an incremental EM algorithm.
- The memory is optimized through the GAN loss rather than through a memory loss.
- The contribution of each sample is decided by tracking the slot histogram.

### Discriminator Output

For each input $x$, we first need to decide which memory slot to use to compute the discriminative probability.

$c \in\{1,2, \ldots, N\}$ denotes the memory slot index.

A Von Mises-Fisher (vMF) mixture model is used to represent the posterior distribution of the memory index.

$$p(c=i | x)=\frac{p(x | c=i) p(c=i)}{\sum_{j=1}^{N} p(x | c=j) p(c=j)}=\frac{\exp \left(\kappa K_{i}^{T} \mu(x)\right) p(c=i)}{\sum_{j=1}^{N} \exp \left(\kappa K_{j}^{T} \mu(x)\right) p(c=j)}$$

In $p(x | c=i)=C(\kappa) \exp \left(\kappa K_{i}^{T} \mu(x)\right)$, $\kappa=1$ is a constant attention parameter.

The vMF is effectively equivalent to a properly normalized Gaussian distribution defined on the unit sphere.

The categorical distribution over the memory index, $p(c)$, is obtained by normalizing the slot histogram, $p(c=i)=\frac{h_{i}+\beta}{\sum_{j=1}^{N}\left(h_{j}+\beta\right)}$, where $\beta\left(=10^{-8}\right)$ is a smoothing constant for numerical stability. By using $p(y=1 | c=i, x)=v_{i}$, we marginalize the joint probability $p(y=1, c | x)$ over c to obtain $p(y=1 | x)$:

$$
p(y=1 | x)=\sum_{i=1}^{N} p(y=1 | c=i, x) p(c=i | x)=\sum_{i=1}^{N} v_{i} p(c=i | x)=\mathbb{E}_{i \sim p(c | x)}\left[v_{i}\right]
$$

However, for each sample x, this is not scalable for the whole memory of size N.

The maximum a posteriori probability is adopted to consider the top-k slots $S=\left\{s_{1}, 
\dots, s_{k}\right\}$:

$$
S=\underset{c_{1}, \ldots, c_{k}}{\operatorname{argmax}} p(c | x)=\underset{c_{1}, \ldots, c_{k}}{\operatorname{argmax}} p(x | c) p(c)=\underset{c_{1}, \ldots, c_{k}}{\operatorname{argmax}} \exp \left(\kappa K_{c}^{T} \mu(x)\right)\left(h_{c}+\beta\right)
$$

where $p(x | c)$ is the vMF likelihood and $p(c)$ is the prior distribution of the memory index.

Here we omit the normalization of the vMF likelihood and the denominator of the prior, because they are both constants. Once we have obtained S, we can compute

$$
p(y | x) \approx \frac{\sum_{i \in S} v_{i} p(x | c=i) p(c=i)}{\sum_{j \in S} p(x | c=j) p(c=j)}
$$

### Memory Update Mechanism

The memory keys and values are updated during training. The update mechanism includes a traditional memory update mechanism and an incremental EM algorithm.

Let the sample be $x$ with label $y$, where real is 1 and fake is 0. For each x, we first find the k-nearest slots $S_{y}$, but using the conditional posterior $p\left(c | x, v_{c}=y\right)$. This is so that in the subsequent EM algorithm, only the slots belonging to the same class as y are considered.

After that, we update the memory in different ways depending on whether $S_y$ contains the correct label.

#### Without the correct label:

The oldest memory slot is found via $n_{a}=\operatorname{argmax}_{i \in\{1, \ldots, N\}} a_{i}$, and the information of x is copied to the corresponding position: $K_{n_{a}} \leftarrow q=\mu(x)$, $v_{n_{a}} \leftarrow y$, $a_{n_{a}} \leftarrow 0$, $h_{n_{a}} \leftarrow \frac{1}{N} \sum_{i=1}^{N} h_{i}$.

#### With the correct label:

The memory keys are updated through the following custom incremental EM algorithm run for T iterations, which partly incorporates the information of the new sample.

In the inference step, by applying the keys of the previous moment $\hat{K}_{i}^{t-1}$ and $\hat{h}_{i}^{t-1}$ to Equation 1, the posterior distribution $\gamma_{i}^{t}=p\left(c_{i} | x\right)$ is computed, where $i \in S_{y}$. In the maximization step, the following updates are performed:

$$
\hat{h}_{i}^{t} \leftarrow \hat{h}_{i}^{t-1}+\gamma^{t}-\gamma^{t-1}, \quad \hat{K}_{i}^{t} \leftarrow \hat{K}_{i}^{t-1}+\frac{\gamma^{t}-\gamma^{t-1}}{\hat{h}_{i}^{t}}\left(q_{i}-\hat{K}_{i}^{t}\right)
$$

where $t \in 1, \ldots, T, \gamma^{0}=0, \hat{K}_{i}^{1}=K_{i}, \hat{h}_{i}^{1}=\alpha h_{i}$, and $\alpha=0.5$.

After T iterations, the slots of $S_y$ are updated through $K_{i} \leftarrow \hat{K}_{i}^{t}$ and $h_{i} \leftarrow \hat{h}_{i}^{t}$.

The decay rate $\alpha$ controls the degree to which the contribution of old queries to the slot position in the average direction of the mixture component is exponentially reduced.

$\alpha$ is crucial for performance, because the old queries used to update the keys no longer fit the current mixture distribution, as the feedforward network itself is also being updated.

Finally, it is worth noting that this memory update mechanism is orthogonal to the adversarial training algorithm, because when the discriminator is updated, the memory update is carried out independently. In addition, adding the memory module does not affect the speed of the model at test time, because the memory is only updated during training.

## Memory Conditional Generative Network

The memory conditional generative network is based on the generator of InfoGAN. The difference is that it is conditioned not only on random noise but also on memory information.

In other words, the generator not only samples randomly from the noise distribution, but also samples the memory index $i$ from $P\left(c=i | v_{c}=1\right)=\frac{h_{i} v_{i}}{\sum_{j}^{N} h_{j} v_{j}}$. The formula above represents the frequency of occurrence of the unit i that stores real data. The final input is $[K_i, z]$, where $K_i$ is the key vector of memory index i.

The difference from other CGANs is that MCGN does not require extra annotations or an extra encoder. Instead, MCGN can make full use of the memory information learned by the DMN in an unsupervised way. The DMN learns the vMF mixture memory using only each sample in the sequence and its label.


The training procedure of the whole MemoryGAN is:

for number of training iterations do
    Sample a minibatch of samples from the training set
    Sample a minibatch from the noise distribution and the memory indices
    Update the discriminator loss
    Find S_y for each sample in the minibatch
    Initialize the key, h and gamma for each slot in S_y
    for number of EM iterations do
        Estimate $\gamma_s$ for each s
        Update $h_s$
        Update $K_s$
    Update the vMF mixture model, $h_{s_{y}} \leftarrow \hat{h}_{s_{y}}^{T}, K_{s_{y}} \leftarrow \hat{K}_{s_{y}}^{T}$ for $s_{y} \in S_{y}$
    Sample a minibatch from the noise distribution and the memory indices
    Update the generator loss

## objective function

The goal of MemoryGAN is based on the objective of InfoGAN, which is to maximize the mutual information between the latent variables and the observed content. (For details, see InfoGAN.)

A mutual information loss is added between $K_i$ and $G(z, K_i)$ to ensure the continuity between the sampled memory information and the structured information of the generated samples:

$$
I\left(K_{i} ; G\left(z, K_{i}\right)\right) \geq H\left(K_{i}\right)-\hat{I}-\log C(\kappa)
$$

where $\hat{I}$ denotes the expectation of the negative cosine similarity, $\hat{I}=-E_{x \sim G\left(z, K_{i}\right)}\left[\kappa K_{i}^{T} \mu(x)\right]$.

To be continued
