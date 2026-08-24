---
title: "Decoupled Neural Interfaces using Synthetic Gradients"
date: 2020-02-18T11:04:59+08:00
draft: false
math: true
categories: [notes]
---

# Decoupled Neural Interfaces using Synthetic Gradients

## Abstract

Training neural networks typically requires pushing the computation graph forward and then backpropagating errors to update the weights. In this sense, all layers of the network are locked, because they must wait for other parts of the network to complete their forward inference and backward pass before they can be updated. In this work, we introduce models of future computation in the network graph, in order to decouple modules and thereby break this constraint. These models use only local information to predict the outcome of a subgraph. In particular, we focus on modeling the error gradients: by replacing the true backpropagated error gradients with modeled synthetic gradients, subgraphs are decoupled and can be updated independently and asynchronously—namely, we realize decoupled neural interfaces. We present results for feed-forward models in which every layer is trained asynchronously; for RNNs, where predicting the future gradients of one layer extends the effective modeling horizon of the RNN; and for hierarchical RNN systems with different timescales. Finally, we show that, beyond predicting gradients, the same framework can also be used to predict inputs, resulting in models that are decoupled in both the forward and backward passes—equivalent to two independent networks that can learn together so as to be composed into a single functional network.

<!-- more -->

## Introduction

Each layer (or module) in a directed neural network can be viewed as a computation step that transforms its input. These modules are connected by directed edges, forming a feed-forward graph that defines the flow of data from the network's input, through each module, to the network's output. A loss is defined on the output to produce errors, which are backpropagated through the network graph to update each module's weights.

This mechanism leads to the following forms of locking:

1. Forward Locking: no module can process its input before the earlier nodes of the forward computation graph have been executed.
2. Update Locking: no module can be updated until all relevant modules have finished executing in the forward graph.
3. Backwards Locking: no module can be updated until all relevant modules have executed both the feed-forward and the backward model—for example, the BP algorithm.

Forward, update, and backwards locking force neural networks to run and update sequentially and synchronously. For simple networks this seems benign, but it becomes a serious problem for large, complex, irregular networks, or network systems running in multiple environments with asynchronous timescales.

For example, consider distributed models in which part of the model is shared and used by many downstream clients. This means that all clients must finish executing and pass their error gradients back to the shared model before it can be updated—which means the training speed of the system is dictated by the slowest client. If training of the network could be parallelized, computation time could be greatly reduced.

The goal of this work is to remove update locking from neural networks. This can be achieved by removing backpropagation. To update the weights $\theta_{i}$ of module i, we approximate the backpropagation function as closely as possible:

$$
\begin{aligned} \frac{\partial L}{\partial \theta_{i}} &=f_{\text {Bprop }}\left(\left(h_{i}, x_{i}, y_{i}, \theta_{i}\right), \ldots\right) \frac{\partial h_{i}}{\partial \theta_{i}} \\ & \simeq \hat{f}_{\text {Bprop }}\left(h_{i}\right) \frac{\partial h_{i}}{\partial \theta_{i}} \end{aligned}
$$

where h denotes the layer's activations, x the input, y the supervision (labels), and L the total loss. The update now depends entirely on h—that is, on information local to module i.

The premise of this method is to allow neural network modules to interact and to be trained without update locking. Here the authors replace the conventional neural interface (the connection between two modules in a network) with a Decoupled Neural Interface (DNI). In short, when one network layer passes activations to another layer, there is an associated model that produces a predicted error gradient for those activations. The predicted gradient is a function of the activation alone and does not depend on other events, states, or losses. The sending layer can then use this synthetic gradient to update immediately. By removing update and backwards locking, we can train networks without synchronization. We also present preliminary results extending this idea to remove forward locking as well, so that the modules of the network can likewise be trained without a synchronized forward pass. When applied to RNNs, we show that using synthetic gradients allows an RNN to model a time horizon far beyond the limit of BPTT. We further show that using synthetic gradients to decouple two RNNs operating at different timescales can greatly improve training.

## Decoupled Neural Interfaces

We first describe the high-level communication protocol used to allow asynchronous learning agents to communicate.

As shown in Figure 1, Sender A sends a message (activation) $h_A$ to Receiver B. Intuitively, A can be thought of as the earlier layer and B as the later layer. B has a utility $M_B$ that processes the signal $h_A$ to predict the feedback. The error signal: $\hat{\delta}_{A}=M_{B}\left(h_{A}, s_{B}, c\right)$, where $h_A$ is the message (activation), $s_B$ is B's state, and c is some other potentially relevant information, such as labels or context. A can update immediately using this error signal. B can also, in time, fully evaluate the true $\delta_A$; thus B's utility model can be updated to fit the true utility, reducing the discrepancy between the true and synthetic errors.

This protocol allows A to send messages to B in a way that decouples the updates of A and B—A does not have to wait for B to evaluate the true utility before updating—and A can still learn to send messages to B.

We can apply this protocol to the communication process within networks, giving rise to so-called Decoupled Neural Interfaces (DNI). For neural networks, the feedback error signal $\hat{\delta}_A$ can take different forms: for example, gradients can serve as error signals used together with backpropagation; target information can serve as error signals used together with target propagation; or they can even be used as values incorporated into a reinforcement learning framework. This paper focuses on differentiable networks trained via backpropagation with gradient-based updates. We therefore concentrate on the produced error gradient as the feedback $\hat{\delta}_A$, called the synthetic gradient.

Notation. We define the function at step i as fi, and the composition of the functions from step i to step j as Fij. The loss at layer i is defined as Li.

### Synthetic Gradient for Feed-Forward Networks

Consider the DNI formulation for feed-forward networks: N layers fi, each taking input hi-1 and producing output hi = fi(hi-1), where h0 = x is the input data. The feed-forward graph of the whole network can be written as F1N. See Figure 3(a)

![20200218134533.png](http://cdn.ereebay.me/hexo/20200218134533.png)

Define the loss function on the network output as L = LN. Each layer fi has parameters thetai, updated by a gradient update rule to minimize L(hN)

$$
\theta_{i} \leftarrow \theta_{i}-\alpha \delta_{i} \frac{\partial h_{i}}{\partial \theta_{i}} ; \quad \delta_{i}=\frac{\partial L}{\partial h_{i}}
$$

α is the learning rate, and the derivative with respect to hi is obtained via backpropagation. The dependence on $\delta_i$ means that layer i's update must wait until the rest of the network has been updated—that is, all of F(i+1)N has executed its feed-forward and feedback phases. Layer i is thus update-locked to F(i+1)N.

To remove update locking, we adopt the communication protocol described earlier. Layer i sends the message hi to the following layer, which has a communication model Mi+1 that produces the synthetic error gradient $\hat{\delta}_{i} = M_{i+1}(h_{i})$. As shown in the figure:

![20200218135141.png](http://cdn.ereebay.me/hexo/20200218135141.png)

Layer i, as well as the other layers within F1i, can be updated immediately

$$
\theta_{n} \leftarrow \theta_{n}-\alpha \hat{\delta}_{i} \frac{\partial h_{i}}{\partial \theta_{n}}, n \in\{1, \ldots, i\}
$$

To train the parameters of the synthetic gradient model Mi+1, we wait until the true error gradient $\delta_i$ has been computed, and then compute the MSE between the two.

Moreover, for feed-forward networks, we can use synthetic gradients as the communication feedback to decouple every layer in the network. As shown in the figure:

![20200218142500.png](http://cdn.ereebay.me/hexo/20200218142500.png)

The complete execution of this mechanism:

![20200218150932.png](http://cdn.ereebay.me/hexo/20200218150932.png)

In this case, the target error gradient $\delta_i$ is produced by backpropagating $\hat{\delta}_{i+1}$ through layer i+1. Thus $\delta_i$ is not the true error gradient but an estimate obtained from the synthetic gradient models further downstream. Surprisingly, this does not make the errors worse—learning remains stable even across many layers, as the experiments below show.

In addition, supervision or context c can be incorporated when computing the synthetic gradient.

$$
\hat{\delta}_{i} = {M_{i+1}\left(h_{i}, c\right) }
$$

This procedure allows a layer to be updated immediately after its forward pass has been executed. This paves the way for training sub-parts—or individual layers—of a network in an asynchronous manner.

## Experiment

### Feed-Forward Networks

We apply DNI to feed-forward networks to allow asynchronous or sporadic training of individual layers, as might arise in distributed training.

As explained earlier, introducing synthetic gradients decouples the network layers, allowing them to communicate with one another without being affected by update locking.

**Asynchronous Updates** To demonstrate the improvements brought by the DNI-decoupled layers, we ran experiments on a four-layer fully connected network for MNIST, in which each layer's backward pass and update occur in random order with probability pupdate. (That is, a layer is updated only for a fraction pupdate of the forward passes.) This completely breaks backpropagation—for example, the first layer would only receive error-gradient updates with probability pupdate³—and even then, the system would still be constrained to be synchronous. With DNI, however, the communication gap between layers is bridged, and the randomness of layer updates does not affect downstream layers, because synthetic gradients are used. We sampled different values of pupdate uniformly between 0 and 1. DNI with and without labels is shown in Figure 7.

![20200218170640.png](http://cdn.ereebay.me/hexo/20200218170640.png)

With pupdate = 0.2, the network can still be trained to 2% error. Incredibly, when DNI is conditioned on the data labels (a reasonable assumption if training is performed in a distributed manner), the network trains perfectly with an update probability of just 5%, albeit more slowly.

### Complete Unlock

By also eliminating forward locking, feed-forward networks are made fully asynchronous. In this case, every network layer has a synthetic gradient model as well as a synthetic input model used to predict the input data, as shown in the figure below:

![20200218175611.png](http://cdn.ereebay.me/hexo/20200218175611.png)

Each network layer can then be trained independently, relying on the synthetic gradient and the synthetic input model. The figure below shows the experimental results.

![20200218175713.png](http://cdn.ereebay.me/hexo/20200218175713.png)

The experiments show that in this setting the model can be trained in a fully asynchronous and independent manner, although it takes somewhat longer to reach the 2% error rate.

## Discussion & Conclusion

This paper introduced DNI, a method that uses synthetic gradients to decouple the communication between network layers, allowing them to update independently. It also demonstrated that the method can fully separate all layers of the network, so that they can train in a completely asynchronous, unordered, and sporadic manner.

It is worth noting that, although this paper presented empirical evidence for the effectiveness of DNI and synthetic gradients, the work of Czarnecki et al. investigated the theoretical understanding in greater depth and established convergence.
