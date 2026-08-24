---
title: "Notes on Generative Adversarial Networks (GAN)"
date: 2019-03-01T17:00:22+08:00
draft: false
math: true
categories: [notes]
tags: [GAN, note, deep learning]
---

# Generative Adversarial Network

## Overview of GAN

The idea behind GAN is a two-player zero-sum game in which the sum of the two players' interests is a constant. Consider arm wrestling: assuming the total space is fixed, if your strength is greater, you gain more space and I get less; conversely, if I am stronger, I gain more. But one thing is certain — our total space is fixed. This is a two-player game in which the total interest is constant.

<!-- more -->

A vivid analogy: GAN is like a large network containing two smaller networks. One is the generative network, which can be regarded as a counterfeiter producing fake banknotes, and the other is the discriminative network, i.e., the person authenticating the banknotes. The goal of the generative network is to fool the discriminator, while the discriminator strives not to be fooled by the generator. Through alternating optimization during training, both networks improve. It is theoretically proven that, at best, the generative model makes the discriminator unable to tell real from fake, i.e., the probability of real versus fake is fifty-fifty.

<!-- ![](../res/img/img1.jpg) -->

The figure above is a structural diagram of a generative adversarial network: the discriminator takes in both real samples and fake samples produced by the generator, and outputs a real/fake verdict. The generator takes noise as input and produces fake samples.

## The Principle of GAN

The following is the objective function of GAN:

$$
\min _ { G } \max _ { D } V ( D , G ) = \mathbb { E } _ { \boldsymbol { x } \sim p _ { \mathrm { data } } ( \boldsymbol { x } ) } [ \log D ( \boldsymbol { x } ) ] + \mathbb { E } _ { \boldsymbol { z } \sim p _ { \boldsymbol { z } } ( z ) } [ \log ( 1 - D ( G ( \boldsymbol { z } ) ) ) ]
$$ 

From the objective function we can see that the overall cost function minimizes the generator and maximizes the discriminator. When tackling this optimization problem, we can first fix G, maximize D, and then minimize G to obtain the optimal solution. Here, for a given G, maximizing V(D,G) measures the divergence or distance between P_G and P_data.

First, after fixing G, the optimal D can be expressed as:

$$
D _ { G } ^ { * } = \operatorname { argmax } _ { D } V ( G , D )
$$

The problem of optimizing G can then be expressed as:

$$
G ^ { * } = \operatorname { argmin } _ { G } V \left( G , D _ { G } ^ { * } \right)
$$

## Theoretical Derivation

The original paper's derivation uses the JS divergence to describe the similarity between two distributions, and the JS divergence is in turn built from the KL divergence. Therefore, before carrying out the full derivation, we first introduce some theoretical foundations, derive the conditions required for the optimal discriminator and the optimal generator, and finally use the derived results to restate the training procedure.

### KL Divergence

For a single random variable $x$ with two separate probability distributions $P(x)$ and $Q(x)$, the KL divergence can measure the difference between the two distributions (the appendix proves why the KL divergence reflects the difference between two distributions):

$$
D _ { \mathrm { KL } } ( P \| Q ) = \mathbb { E } _ { \mathrm { x } \sim P } \left[ \log \frac { P ( x ) } { Q ( x ) } \right] = \mathbb { E } _ { \mathrm { x } \sim P } [ \log P ( x ) - \log Q ( x ) ]
$$

Properties of the KL divergence:

1. Non-negativity (used later in the derivation); moreover, the KL divergence equals 0 if and only if P and Q are the same distribution. Because of this non-negativity, it is often used to measure the difference between two distributions. (The appendix proves its non-negativity.)
2. Asymmetry. Although it can measure the difference between distributions, this difference is not a symmetric distance: the KL divergence of P with respect to Q differs from that of Q with respect to P.

### Issues in the Paper's Derivation

In the original paper, one idea differs from many other approaches: the generator G is not required to satisfy an invertibility condition, and in practice G is indeed non-invertible. However, in the proof, the change-of-variables formula for integrals was incorrectly used, while this change of variables is valid only when G is invertible. So the proof should be based on the validity of the following equality:

$$
E _ { z \sim p _ { x } ( z ) } \log ( 1 - D ( G ( z ) ) ) = E _ { x \sim p _ { c } ( x ) } \log ( 1 - D ( x ) )
$$

This equality comes from the Radon-Nikodym theorem in measure theory; it is presented as Proposition 1 in the original paper and expressed as the following equality:

$$
\begin{array} { c } { \int _ { x } p _ { d a t a } ( x ) \log D ( x ) \mathrm { d } x + \int _ { z } p ( z ) \log ( 1 - D ( G ( z ) ) ) \mathrm { d } z } \\ { = \int _ { x } p _ { d a t a } ( x ) \log D ( x ) + p _ { G } ( x ) \log ( 1 - D ( x ) ) \mathrm { d } x } \end{array}
$$

This formula uses the change-of-variables formula for integrals, but doing so requires computing the inverse of G, whose existence is not assumed. Moreover, in the practice of neural networks it does not exist. However, this practice is so common in ML that it is simply overlooked.

### Optimal Discriminator

In the minimax game, we first fix the generator G and maximize the value function, which yields the optimal discriminator D. The maximized value function measures the divergence between the distribution produced by the generator and the distribution of the dataset (proven later).

The expectations in the original paper's value function can be expanded into integral form:

$$
V ( G , D ) = \int _ { x } p _ { \mathrm { data } } ( \boldsymbol { x } ) \log ( D ( \boldsymbol { x } ) ) + p _ { g } ( \boldsymbol { x } ) \log ( 1 - D ( \boldsymbol { x } ) ) d x
$$

Maximizing the integral can be converted into maximizing the integrand. Maximizing the integrand yields the optimal discriminator D, so all terms not involving the discriminator can be treated as constants. Letting the discriminator D(x) be y, the integrand can be written as:

$$
f ( y ) = a \log y + b \log ( 1 - y )
$$

To find the optimal extreme point, if $a + b \neq 0$, we can solve using the first-order derivative:

$$
f ^ { \prime } ( y ) = 0 \Rightarrow \frac { a } { y } - \frac { b } { 1 - y } = 0 \Rightarrow y = \frac { a } { a + b }
$$

Taking the second derivative at the stationary point gives:

$$
f ^ { \prime \prime } \left( \frac { a } { a + b } \right) = - \frac { a } { \left( \frac { a } { a + b } \right) ^ { 2 } } - \frac { b } { 1 - \left( \frac { a } { a + b } \right) ^ { 2 } } < 0
$$

where $a , b \in ( 0,1 )$. Since the first derivative equals 0 and the second derivative is less than 0, $\frac { a } { a + b }$ is a maximum.

Finally, the value function can be written as:

$$
\begin{aligned} V ( G , D ) = & \int _ { x } p _ { d a t a } ( x ) \log D ( x ) + p _ { G } ( x ) \log ( 1 - D ( x ) ) \mathrm { d } x \\ & \leq \int \max _ { y } \max _ { y } p _ { d a t a } ( x ) \log y + p _ { G } ( x ) \log ( 1 - y ) \mathrm { d } x \end{aligned}
$$

Letting D(x)=P_data/(P_data+p_G) attains the maximum, because f(y) has a unique maximum on its domain; that is, the optimal D is unique, and no other D can attain the maximum.

In fact, the optimal D cannot be computed in practice, but it is mathematically important. Moreover, we do not know the prior Pdata, so we cannot use it directly in training. On the other hand, the existence of the optimal D implies that of the optimal G, and it suffices that we approach the optimal D.

### Optimal Generator

The training process of GAN aims to make P_G=P_data, in which case the optimal D can be written as:

$$
D _ { G } ^ { * } = \frac { p _ { \text {data} } } { p _ { \text {data} } + p _ { G } } = \frac { 1 } { 2 }
$$

That is, the optimal generator makes the discriminator unable to distinguish P_data from P_G. Based on this observation, the authors proved that G is the solution of the minimax game.

Theorem: The global minimum of C(G)=maxV(G,D) is attained if and only if P_G=P_data.

The theorem states an if-and-only-if result, so we prove it from both directions. First we work backwards to derive the value of C(G), then we prove it forward.

Assume P_G=P_data (deriving backwards from the known result). We can derive:

$$
V \left( G , D _ { G } ^ { * } \right) = \int _ { x } p _ { d a t a } ( x ) \log \frac { 1 } { 2 } + p _ { G } ( x ) \log \left( 1 - \frac { 1 } { 2 } \right) \mathrm { d } x
$$

$$
V \left( G , D _ { G } ^ { * } \right) = - \log 2 \int _ { x } p _ { G } ( x ) \mathrm { d } x - \log 2 \int _ { x } p _ { d a t a } ( x ) \mathrm { d } x = - 2 \log 2 = - \log 4
$$

So -log4 is a candidate for the minimum, since it arises only when P_G=P_data. Now we must prove forward that this value is always the minimum, i.e., that the if-and-only-if condition is fully satisfied.

Now drop the condition P_G=P_data, pick any G, and rewrite the formula as:

$$
C ( G ) = \int _ { x } p _ { d a t a } ( x ) \log \left( \frac { p _ { d a t a } ( x ) } { p _ { G } ( x ) + p _ { d a t a } ( x ) } \right) + p _ { G } ( x ) \log \left( \frac { p _ { G } ( x ) } { p _ { G } ( x ) + p _ { d a t a } ( x ) } \right) \mathrm { d } x
$$

Next we apply a trick: add a zero to the equation. It does not change the value of the equation, but it lets us construct a log2, since we know -log4 is the candidate global minimum.

$$
\begin{aligned} C ( G ) & = \int _ { x } ( \log 2 - \log 2 ) p _ { d a t a } ( x ) + p _ { d a t a } ( x ) \log \left( \frac { p _ { d a t a } ( x ) } { p _ { G } ( x ) + p _ { d a t a } ( x ) } \right) \\ & + ( \log 2 - \log 2 ) p _ { G } ( x ) + p _ { G } ( x ) \log \left( \frac { p _ { G } ( x ) } { p _ { G } ( x ) + p _ { d a t a } ( x ) } \right) \mathrm { d } x \end{aligned}
$$

$$
\begin{array} { c } { C ( G ) = - \log 2 \int _ { x } p _ { G } ( x ) + p _ { d a t a } ( x ) d x } \\ { + \int _ { x } p _ { d a t a } ( x ) \left( \log 2 + \log \left( \frac { p _ { d a t a } ( x ) } { p _ { G } ( x ) + p _ { d a t a } ( x ) } \right) \right) } \\ { + p _ { G } ( x ) \left( \log 2 + \log \left( \frac { p _ { G } ( x ) } { p _ { G } ( x ) + p _ { d a t a } ( x ) } \right) \right) \mathrm { d } x } \end{array}
$$

Finally, simplification yields:

$$
\begin{aligned} C ( G ) = & - \log 4 + \int _ { x } p _ { d a t a } ( x ) \log \left( \frac { p _ { d a t a } ( x ) } { \left( p _ { G } ( x ) + p _ { \text {data} } ( x ) \right) / 2 } \right) \mathrm { d } x \\ & + \int _ { x } p _ { G } ( x ) \log \left( \frac { p _ { G } ( x ) } { \left( p _ { G } ( x ) + p _ { d a t a } ( x ) \right) / 2 } \right) \mathrm { d } x \end{aligned}
$$

If you have read the earlier section on KL divergence, you will notice that this can be simplified into the form of KL divergences:

$$
C ( G ) = - \log 4 + K L \left( p _ { d a t a } | \frac { p _ { d a t a } + p _ { G } } { 2 } \right) + K L \left( p _ { G } | \frac { p _ { d a t a } + p _ { G } } { 2 } \right)
$$

Since the KL divergence is non-negative, -log4 is the global minimum.

It remains to show that only one G can attain this value, so that P_G=P_data is the unique solution, completing the proof.

From the earlier discussion we know the KL divergence is asymmetric and can only measure the similarity of distribution a with respect to distribution b. But after adding the second term, their sum becomes symmetric, and this sum of the two terms can be expressed as the JS divergence:

$$
\operatorname { JSD } ( P \| Q ) = \frac { 1 } { 2 } D ( P \| M ) + \frac { 1 } { 2 } D ( Q \| M )
$$

$$
M = \frac { 1 } { 2 } ( P + Q )
$$

Suppose there are two distributions P and Q whose average distribution is M=(P+Q)/2; then the JS divergence between the two distributions is the KL divergence between P and M plus the KL divergence between Q and M, divided by 2.

Therefore, the JS divergence ranges from 0 to log2. It equals log2 when the two distributions have no overlap at all, and reaches the minimum 0 when they are identical.

Hence C(G) can be rewritten as:

$$
C ( G ) = - \log 4 + 2 \cdot J S D \left( p _ { \text { data } } | p _ { G } \right)
$$

This proves that the JSD is 0 when P_G=P_data. In summary, we obtain the optimal generator if and only if the generated distribution equals the true data distribution.

### Convergence

Regarding whether the training process converges to the optimal generator, the original paper gives an additional proof that, with sufficient training data and sufficient capacity of d and g, training converges to the optimal G. Since this part is not particularly important, I will put the proof in the appendix (mainly because my own understanding of it is only partial).

### Training Procedure

1. Parameter optimization process

If we want to find the optimal generator, then after fixing a discriminator D, we can treat the original value function as the loss function L(G) for training the generator. With a loss function, we can update the generator using optimization algorithms such as SGD or Adam. The gradient descent update is as follows:

$$
\theta _ { G } \leftarrow \theta _ { G } - \eta \partial L ( G ) / \partial \theta _ { G }
$$

Now, given an initial G_0, we need to find the D_0* that maximizes V(G_0,D); thus the discriminator's update process amounts to minimizing the loss function -V(G,D). Moreover, from the earlier derivation, V(G,D) differs from the JS divergence between the distributions P_data(x) and P_G(x) only by a constant term. Therefore this alternating adversarial process can be described as:

- Given G_0, maximize V(G_0,D) to obtain D_0*, i.e., max[JSD(P_data(x)||P_G0(x))];
- Fix D_0*, compute $\mathrm { \theta } _ { - } \mathrm { G } 1 \leftarrow \theta _ { - } \mathrm { G0 } - \mathrm { \eta } \left( \partial \mathrm { V } \left( \mathrm { G } , \mathrm { D } _ { - } \mathrm { 0 } ^ { * } \right) / \partial \theta _ { - } \mathrm { G } \right)$ to obtain the updated G_1;
- Fix G_1, maximize V(G_1,D_0*) to obtain D_1*, i.e., max[JSD(P_data(x)||P_G1(x)];
- Fix D_1*, compute $\theta _ { - } \mathrm { G } 2 \leftarrow \theta _ { - } \mathrm { G } 1 - \eta \left( \partial \mathrm { V } \left( G , D _ { - } \mathrm { 0 } ^ { * } \right) / \partial \theta _ { - } \mathrm { G } \right)$ to obtain the updated G_2;

And so the loop continues.

2. Practical training process

According to the definition of the value function V(G,D) above, we need to compute two expectations, E[log(D(x))] and E[log(1-D(G(z)))], where x follows the real data distribution and z follows the initialization distribution. In practice, there is no way to compute these two expectations by integration, so we generally approximate them by sampling from the infinite real data and an infinite generator.

Suppose the generator G is given and we wish to compute maxV(G,D) to find the discriminator D. First we need to sample m samples from P_data(x) and m samples from the generator P_G(x). Maximizing the value function can then be replaced by the following expression:

$$
\text { Maximize } \tilde { V } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \log D \left( x ^ { i } \right) + \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \log \left( 1 - D \left( \tilde { x } ^ { i } \right) \right)
$$

Now we treat the samples drawn from P_data(x) as positive samples and those drawn from P_G(x) as negative samples, while using an approximation of the negative V(G,D) as the loss function. This can therefore be formulated as the standard training procedure of a binary classifier:

$$
\text { Minimize } L = - \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \log D \left( x ^ { i } \right) - \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \log \left( 1 - D \left( \tilde { x } ^ { i } \right) \right)
$$

In practice, we must implement the minimax game with iterative and numerical methods. Fully optimizing D in the inner loop of training is computationally prohibitive, and a finite dataset would lead to overfitting. Therefore, we can alternate between k steps of optimizing D and one step of optimizing G. As long as G is updated slowly, D stays near its optimal solution.

In summary, over the whole training procedure, for each iteration:

- Draw m samples from the real data distribution P_data
- Draw m noise samples from the prior distribution P_prior(z)
- Feed the noise samples into the generator G to produce data $\left\{ \tilde { x } ^ { 1 } , \tilde { x } ^ { 2 } , \ldots , \tilde { x } ^ { m } \right\} , \tilde { x } ^ { i } = G \left( z ^ { i } \right)$, and update the discriminator parameters $\theta _ { - } d$ by maximizing the approximation of V; the discriminator's update rule is $\theta _ { d } \leftarrow \theta _ { d } + \eta \nabla \tilde { V } \left( \theta _ { d } \right)$

The above is the process of learning the discriminator D. The process computes the JS divergence and is repeated k times, because we want to maximize the value function.

- Draw another m noise samples from the prior distribution P_prior(z)
- Update the generator by minimizing V, i.e., $\tilde { V } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \log \left( 1 - D \left( G \left( z ^ { i } \right) \right) \right)$, with the generator's update rule $\theta _ { g } \leftarrow \theta _ { g } - \eta \nabla \tilde { V } \left( \theta _ { g } \right)$

The above is the learning process of the generator's parameters. This process occurs only once per iteration, which avoids excessive updates that would make the JS divergence rise.

This concludes the complete derivation and argumentation of GAN.

## Several Issues in GAN Training

### Training Instability

Training the original GAN is very difficult. This mainly manifests as the training process failing to converge, or the trained generator being unable to produce meaningful content, among other issues. On the other hand, although our optimization objective is the JS divergence, which should reflect the distance between the two distributions, ideally this distance should be relatively large at the start and gradually shrink as the training of G progresses.

In practice, the discriminator's loss function very easily drops to 0 and then stays at 0 thereafter. The JS divergence measures the distance between two distributions, but in practice two situations can cause the JS divergence to deem the distance between the two distributions infinite, making the loss function forever 0.

Case 1: the discriminator D is too strong, causing overfitting.

Solution: try regularization, or reduce the number of model parameters.

Case 2: properties of the data itself. The low-dimensional manifold produced by the generator indeed does not overlap easily with the data distribution.

Solution: one option is to add noise to the data so that the generator's distribution and the real data distribution overlap more easily.
Another is the GAN to be discussed next time.

### Mode Collapse

All outputs are the same! This phenomenon is known as Mode Collapse. A likely cause is that the real data has large probability mass in many places in the space, but our generative model has not directly learned the characteristics of the real distribution. To guarantee minimal loss, it would rather always produce the same but certainly correct output than try other, different but possibly wrong outputs. In other words, our generator sometimes cannot cover all internal modes of the data distribution and conservatively picks just one mode that is certainly correct.

## Summary

- GAN combines a generative model with a discriminative model, eliminating the difficulty of defining a loss function for generative models
- It operates on probability distributions and is not constrained by the dimensionality of the generation
- It can be used for semi-supervised learning
