---
title: "Federated Adversarial Domain Adaptation Paper Notes (Incomplete)"
date: 2019-11-24T14:04:02+08:00
draft: false
categories: [notes]
---

# Federated Adversarial Domain Adaptation

Federated learning currently has the problem of knowledge domain shift, which prevents it from having good generalization ability. Domain shift occurs when the labeled data of the source nodes and the unlabeled data of the target node are different.

This paper mainly proposes a way to address the use of knowledge domains in federated learning, so that the knowledge learned by different nodes can be aligned with the data distribution of the target node.

The paper's method mainly applies adversarial adaptation techniques to federated learning. In addition, a dynamic attention mechanism is designed, and feature disentanglement is used to enhance knowledge transfer.

<!-- more -->

## Introduction

Traditional federated learning has a problem: since the data on each node is collected in a non-IID manner, a domain shift problem arises. For example, the images taken by one device are mostly indoor scenes, while those taken by another are mostly outdoor scenes. The method proposed in this paper is: **transferring the knowledge in the data of decentralized nodes to a node of a new and different data domain, without requiring additional user supervision.** This method is also known as Unsupervised Federated Domain Adaptation.

There are currently many unsupervised domain adaptation methods, but the federated learning setting gives rise to the following problems:

1. Data is stored locally and cannot be shared.
2. Model parameters are trained independently on different nodes and converge at different rates, and the contribution to the global model depends on the correlation between the two domains.
3. Since the knowledge learned from the source nodes is highly integrated and difficult to disentangle, it may cause negative transfer.

The method proposed in this paper is named Federated Adversarial Domain Adaptation (FADA), which uses adversarial techniques to solve the domain shift problem in federated learning systems.

The main steps of the method are:

1. First, the federated domain adaptation problem is analyzed from a theoretical perspective, and a general, universal bound is provided.
2. Inspired by the theoretical results, an effective adaptation algorithm is proposed, based on adversarial adaptation and representation disentanglement techniques applied to the federated learning setting.
3. A dynamic attention model is designed to cope with the constantly changing convergence rates in federated learning systems.

![png1](http://cdn.ereebay.me/FADAfada-1.png)

## Related Work

**Unsupervised Domain Adaptation**: The goal of UDA is to transfer knowledge from a labeled data domain to an unlabeled data domain.

**Federated Learning**: I will not go into detail here.

**Feature Disentanglement**: The features extracted by neural networks through complex hidden layers are usually highly integrated. Therefore, learning disentangled features can help remove irrelevant or domain-specific knowledge, and then model the needed domain knowledge.

## Generalization Bound for FDA

1. First, recall the definition of the theoretical error bound for the adaptation problem in the single-data-source setting.
2. Then describe the definition of the error bound in the unsupervised federated domain adaptation setting.

## Federated Adversarial Domain Adaptation

![](http://cdn.ereebay.me/flpaper-2.png)

From the theory in the previous section, it is not difficult to see the importance of the weight $\alpha$ and the distance. The method of this paper learns the weights by proposing a dynamic attention mechanism, and reduces the distance between the source domains and the target domain through a federated adversarial alignment mechanism. In addition, a feature disentanglement method is adopted to extract domain-invariant features to strengthen knowledge transfer.

**Dynamic Attention Mechanism**:

The principle of this mechanism is mainly to increase the weights of nodes with high contributions and decrease the weights of nodes with low contributions. So how do we judge the degree of a node's contribution? This paper uses the gap statistics method to evaluate how well the target features can be clustered.

$$
I=\sum_{r=1}^{k} \frac{1}{2 n_{r}} \sum_{i, j \in C_{r}}\left\|f_{i}^{t}-f_{j}^{t}\right\|_{2}
$$

Assume there are $C_{1}, C_{2}, \ldots, C_{k}$ clusters, where $C_r$ denotes the indices of the objects in the r-th cluster, and $n_{r}=\left|C_{r}\right|$.

Intuitively, a smaller gap statistics value indicates that the feature distribution has smaller intra-class variance. The degree of contribution of each source node is measured by computing the difference between the gap statistics of two iterations.

$$
I_{i}^{g a i n}=I_{i}^{p-1}-I_{i}^{p}
$$

It represents how much the clusters can be improved after the target domain is updated with a source domain's gradient. For the gradient updates, the following is used to decide how much each gradient contributes:

$$
\text { Softmax }\left(I_{1}^{\text {gain }}, I_{2}^{\text {gain }}, \ldots, I_{N}^{\text {gain}}\right)
$$



**Federated Adversarial Alignment**:

Machine learning models can suffer greatly degraded performance due to domain distance. To solve this problem, this paper improves the traditional adversarial training method in the federated learning setting. In the federated learning context, this paper divides the adversarial alignment optimization process into two steps: 1. For each domain, train a local feature extractor, with $G_i$ and $G_t$ corresponding to $D_i$ and $D_t$ respectively. 2. For each ($D_i, D_t$) source-target domain pair, train an adversarial domain discriminator DI to align the two distributions in an adversarial learning way. First, train DI to recognize which feature domain a feature comes from; then train the generator, i.e., the feature extractors ($G_i, G_t$), to fool DI. Note: D can only access the outputs of $G_i$ and $G_t$.

Given the i-th source data domain $\mathbf{X}^{S_{i}}$ and the target domain $\mathbf{X}^T$, the optimization objective of $D I_{i}$ is:

$$
\underset{\Theta^{D I_{i}}{L}}{L_{a d v_{D I}}\left(\mathbf{X}^{S_{i}}, \mathbf{X}^{T}, G_{i}, G_{t}\right)=-\mathbb{E}_{\mathbf{x}^{s_{i} \sim \mathbf{X}^{s_{i}}}}\left[\log D I_{i}\left(G_{i}\left(\mathbf{x}^{s_{i}}\right)\right)\right]-\mathbb{E}_{\mathbf{x}^{t} \sim \mathbf{x}^{t}}\left[\log \left(1-D I_{i}\left(G_{t}\left(\mathbf{x}^{t}\right)\right)\right)\right]}
$$

Then, keeping D fixed, the objective function for updating G is:

$$
\underset{\Theta^{G}{G}_{i}, \Theta^{G_{t}}}{L}\left(\mathbf{X}^{S_{i}}, \mathbf{X}^{T}, D I_{i}\right)=-\mathbb{E}_{\mathbf{x}^{s_{i} \sim \mathbf{X}^{s_{i}}}\left[\log D I_{i}\left(G_{i}\left(\mathbf{x}^{s_{i}}\right)\right)\right]-\mathbb{E}_{\mathbf{x}^{t} \sim \mathbf{X}^{t}}\left[\log D I_{i}\left(G_{t}\left(\mathbf{x}^{t}\right)\right)\right]}
$$

**Feature Disentanglement**:

Under the adversarial learning framework, this paper also adopts adversarial disentanglement to extract domain-invariant features. The paper argues that the extracted features can be divided into domain-specific features and domain-invariant features. As shown in the first figure, the feature disentangler D divides the features into two categories. First, a K-class classifier and a K-class category recognizer are trained to predict labels from the features.

$$
\begin{array}{c}{L_{cross entropy}} \\ {\Theta^{G_i}, \Theta^{D_i}, \Theta^{C_i}, \Theta^{CI_i}}\end{array}=-\mathbb{E}_{\left(\mathbf{x}^{s_i}, \mathbf{y}^{s_i}\right) \sim \widehat{\mathcal{D}}_{s_i}} \sum_{k=1}^{K} \mathbb{1}\left[k=\mathbf{y}^{s_{i}}\right] \log \left(C_{i}\left(f_{d i}\right)\right)-\mathbb{E}_{\left(\mathbf{x}^{s} i, \mathbf{y}^{s} i\right) \sim \widehat{\mathcal{D}}_{s_{i}}} \sum_{k=1}^{K} \mathbb{1}\left[k=\mathbf{y}^{s_{i}}\right] \log \left(C I_{i}\left(f_{d s}\right)\right)
$$

Then, with the category classifier fixed, only the feature disentangler is trained, producing domain-specific features to fool the category classifier.

$$
\underset{\Theta^{D} i, \Theta^{G_{i}}}{L_{e n t}}=-\frac{1}{N_{s_{i}}} \sum_{j=1}^{N_{s_{i}}} \log C I_{i}\left(f_{d s}^{j}\right)=-\frac{1}{N_{s_{i}}} \sum_{j=1}^{N_{s_{i}}} \log C I_{i}\left(D_{i}\left(G_{i}\left(\mathbf{x}^{s_{i}}\right)\right)\right)
$$

The feature disentanglement method transfers knowledge by keeping the class-invariant features and removing the class-specific features. To enhance the disentanglement effect, this paper minimizes the mutual information between the two.

Mutual information is defined as $I\left(f_{d i} ; f_{d s}\right)=\int_{\mathcal{P} \times \mathcal{Q}} \log \frac{d \mathbb{P}_{\mathcal{P} Q}}{d \mathbb{P}_{\mathcal{P}} \otimes \mathbb{P}_{\mathcal{Q}}} d \mathbb{P}_{\mathcal{P} \mathcal{Q}}$, where $\mathbb{P}_{\mathcal{P} \mathcal{Q}}$ is the joint probability distribution of the two kinds of features, and $\mathbb{P}_{\mathcal{P}}=\int_{\mathcal{P}} d \mathbb{P}_{\mathcal{P} \mathcal{Q}}, \mathbb{P}_{\mathcal{Q}}=\int_{\mathcal{Q}} d \mathbb{P}_{\mathcal{P} \mathcal{Q}}$ are the corresponding marginal probability distributions.

The Mutual Information Neural Estimator (MINE) is used to estimate the mutual information:

$$
T_{\theta}: \widehat{I(\mathcal{P} ; \mathcal{Q})}_{n}=\sup _{\theta \in \Theta} \mathbb{E}_{\mathbb{P}_{\mathcal{P} Q}^{(n)}}\left[T_{\theta}\right]-\log \left(\mathbb{E}_{\mathbb{P}_{P}^{(n)} \otimes \mathbb{P}_{Q}^{(n)}}\left[e^{T_{\theta}}\right]\right)
$$

In practice, it can be computed by:

$$
I(\mathcal{P} ; \mathcal{Q})=\iint \mathbb{P}_{\mathcal{P} \mathcal{Q}}^{n}(p, q) T(p, q, \theta)-\log \left(\iint \mathbb{P}_{\mathcal{P}}^{n}(p) \mathbb{P}_{\mathcal{Q}}^{n}(q) e^{T(p, q, \theta)}\right)
$$

To avoid computing integrals, the Monte Carlo method is adopted:

$$
I(\mathcal{P}, \mathcal{Q})=\frac{1}{n} \sum_{i=1}^{n} T(p, q, \theta)-\log \left(\frac{1}{n} \sum_{i=1}^{n} e^{T\left(p, q^{\prime}, \theta\right)}\right)
$$

where (p, q) are sampled from the joint distribution, $q^{\prime}$ is sampled from the marginal distribution, and $T(p, q, \theta)$ is a neural network determined by $\theta$ that estimates the mutual information between the P and Q distributions.

The domain-invariant features and the domain-specific features are fed into a reconstructor to obtain recombined features; the loss function uses an L2 loss to reconstruct the original features, in order to preserve the composability of the representation. The L2 constraint and the mutual information loss constraint can be balanced by tuning the hyperparameters.


To be continued
