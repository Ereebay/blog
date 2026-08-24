---
title: "Federated Meta-Learning with Fast Convergence and Efficient Communication"
date: 2020-02-05T18:36:17+08:00
draft: false
math: true
categories: [notes]
---

# Federated Meta-Learning with Fast Convergence and Efficient Communication

## Abstract

This paper proposes a federated meta-learning framework, FedMeta, which shares a parameterized algorithm (meta learner) instead of the previous global model. It is evaluated on the LEAF datasets and a real-world dataset, and demonstrates that the communication cost required by FedMeta is reduced by 2.82-4.33 times, that it converges faster, and that accuracy is even improved by 3.23~14.84 percentage points compared with FedAvg. Moreover, FedMeta preserves user privacy, since only the parameterized algorithm is shared and no data.

<!-- more -->

## Introduction

The well-known FedAvg algorithm in federated learning can flexibly use SGD for local training, achieving high accuracy while balancing computation and communication costs.

In meta-learning, initialization-based meta-learning algorithms such as MAML excel at fast convergence on new tasks and possess good generalization. This makes them well suited to decentralized data that is Non-IID and highly personalized.

Inspired by this, the paper develops a federated meta-learning framework that differs greatly from previous federated learning work. It first connects meta-learning methods with federated learning. In meta-learning, a parameterized algorithm (meta learner) is slowly learned from a large number of tasks through the meta-training process, during which the algorithm quickly trains a specific model within each task.

A task usually consists of a support set and a query set, which are disjoint. The task-specific model is trained on the support set and tested on the query set. The test results are then used to update the algorithm. In federated meta-learning, the algorithm (meta learner) is maintained on the server and distributed to the clients for model training. During each episode of meta-training, a sampled batch of clients receives the parameters of the algorithm (meta learner) and trains their models. The test results on the query sets are then uploaded to the server to update the meta learner. The overall flow is shown in the figure:

![20200218221422.png](http://cdn.ereebay.me/hexo/20200218221422.png)

**Comparing federated meta-learning with federated learning** A comparison of FML and FL.
Federated meta-learning is similar to federated learning; the main difference is that what is shared is no longer a global model but the parameters of an algorithm (the meta-model parameters). Moreover, meta-learning is conceptually different from distributed model training, and sharing a meta-learning algorithm can be applied more flexibly than sharing a model. For example, in image classification, images of n classes may be unevenly distributed across clients, with each client having at most k classes. Federated learning would need to train a large n-class classifier in order to exploit data from all clients, whereas in fact a k-class classifier suffices, since it only ever makes predictions for a single client. Such a large model in federated learning entails substantial communication cost. One could send only part of the model to a client to update the relevant parameters, but this requires prior knowledge of the client's private data. Meta-learning, on the other hand, allows an algorithm to train on tasks containing different classes. For example, MAML can provide an initialization for a k-class classifier by meta-training on k-class tasks, regardless of what the specific classes are. Therefore, within the FML framework, MAML can be used to meta-train initializations of k-class classifiers over all n classes—that is, letting a k-class classifier be meta-learned for initialization in the context of tasks drawn from n classes. This reduces both the communication and computation costs of FML.

**Contributions** The contributions of this paper focus on algorithm design for the federated learning setting; to this end, the paper proposes a new framework and carries out extensive experiments. The contributions are threefold: 1. Proposing the FedMeta framework, which combines meta-learning and federation, integrating MAML and Meta-SGD into federated learning. 2. Conducting experiments on the LEAF datasets with comparisons against FedAvg; the results show that FedMeta reduces overhead while achieving higher accuracy. 3. Applying FedMeta to a recommendation task in which each client has highly personalized records; the experiments demonstrate that meta-learning algorithms achieve higher accuracy than standalone or federated learning methods.

## Federated Meta-Learning

### The Meta-Learning Approach

The goal of meta-learning is to meta-train an algorithm (meta learner) A that can train a model quickly—that is, to obtain an initial model that converges fast. The algorithm $A_{\varphi}$ is a parameterized model whose parameters are updated across a series of tasks during meta-training.
During meta-training, a task T consists of a support set and a query set, each with corresponding labels and data. The algorithm (meta learner) A trains the model f on the support set and outputs the parameters $\theta_T$—this is the inner update. The model $f_{\theta_T}$ is then evaluated on the query set, and the test loss $\mathcal{L}_{D_{Q}^{T}}\left(\theta_{T}\right)$ is computed to reflect how well algorithm A trains. Finally, A is updated by minimizing the test loss—this step is called the outer update. Note that the query set and the support set are disjoint, so as to maximize the generalization ability of A. During meta-training, each episode samples a batch of tasks from a meta-training set. The optimization objective of algorithm A can thus be expressed as:

$$
\min _{\varphi} \mathbb{E}_{T \sim \mathcal{T}}\left[\mathcal{L}_{D_{Q}^{T}}\left(\theta_{T}\right)\right]=\min _{\varphi} \mathbb{E}_{T \sim \mathcal{T}}\left[\mathcal{L}_{D_{Q}^{T}}\left(\mathcal{A}_{\varphi}\left(D_{S}^{T}\right)\right)\right]
$$

MAML is a representative meta-learning algorithm. For MAML, algorithm A is what produces the model's initial state. Concretely, for each task T, the algorithm sets $\alpha = \theta$, making the algorithm's parameters equal to the parameters of model f. The parameters of model f are then trained on the support set and updated according to the loss: $\mathcal{L}_{D_{S}^{T}}(\theta):=\frac{1}{\left|D_{S}^{T}\right|} \sum_{(x, y) \in D_{S}^{T}} \ell\left(f_{\theta}(x), y\right)$. Finally, the model parameters are tested on the query set, and the test loss is computed: $\mathcal{L}_{D_{Q}^{T}}\left(\theta_{T}\right):=\frac{1}{\left|D_{Q}^{T}\right|} \sum_{\left(x^{\prime}, y^{\prime}\right) \in D_{Q}^{T}} \ell\left(f_{\theta_{T}}\left(x^{\prime}\right), y^{\prime}\right)$

Equation 1 can be simplified as:

$$
\min _{\theta} \mathbb{E}_{T \sim \mathcal{T}}\left[\mathcal{L}_{D_{Q}^{T}}\left(\theta-\alpha \nabla \mathcal{L}_{D_{S}^{T}}(\theta)\right)\right]
$$

Building on MAML, Meta-SGD further learns the initial parameters and the inner-loop learning rate at the same time. The test loss can be viewed as a function of both the model parameters and the learning rate, and both can be updated by taking gradients with SGD in the outer loop. Moreover, the learning rate is a vector of the same dimension as the model parameters, so that the learning rate corresponds to the model parameter vector coordinate-wise.

Therefore the optimization condition can be rewritten as

$$
\min _{\theta, \alpha} \mathbb{E}_{T \sim \mathcal{T}}\left[\mathcal{L}_{D_{Q}^{T}}\left(\theta-\alpha \circ \nabla \mathcal{L}_{D_{S}^{T}}(\theta)\right)\right]
$$

### The Federated Meta-Learning Framework

The goal of FML is to meta-train an algorithm collaboratively from data distributed across clients. Taking MAML as an example, the aim is to use the data of all clients to complete the initialization of a model. MAML involves two levels of optimization: the inner loop trains a task-specific model from the initial parameters, and the outer loop updates the initial parameters according to the test loss. In the federated learning setting, each client u receives the initial parameters $\theta$ from the server, then trains the model on the support-set data on its device and transmits the test loss to the server. The server mainly maintains the initial parameters and updates them according to the clients' test losses.

The information transmitted in this process includes: the model's initial parameters (from server to clients) and the test loss (from clients to server). For the Meta-SGD algorithm, the vector $\alpha$ is also part of the transmission, serving as the learning-rate parameters for inner-loop model training.

![20200219111333.png](http://cdn.ereebay.me/hexo/20200219111333.png)

Algorithm 1 describes the FedMeta procedure using MAML and Meta-SGD, where each communication round corresponds to an episode. The algorithm is maintained in the AlgorithmUpdate step. In each round of updates, the server collects the test losses obtained by training MAML or Meta-SGD on the sampled clients. The initial parameters $\theta$ are updated after training on the clients' training sets, and the updated parameters are then tested on the test set. After the meta-training process, the model is deployed on the clients.

## Experiments

1. Experiments on the LEAF datasets demonstrate fast convergence and high accuracy.
2. Experiments on a recommendation task in a real-world setting show that strong adaptability is maintained even at smaller scales.

### Evaluation Scheme

In all experiments, 80% of the clients serve as training clients, 10% as validation clients, and the remainder as test clients. Each client's local data is split into a support set and a query set. The paper also varies the fraction p of each client's data used as the support set, in order to study how effectively FedMeta adapts to new clients with limited data; this is abbreviated psupport hereafter.

![20200219114337.png](http://cdn.ereebay.me/hexo/20200219114337.png)

The authors ran experiments with FedAvg, a meta version of FedAvg, and two variants of FedMeta. The meta version of FedAvg uses the support sets of the test clients to finetune the initial model received from the server before testing, whereas during training both use all of the data on the training clients.

As for FML, three methods were tested—MAML, FOMAML, and Meta-SGD—all of which are model-agnostic and easy to implement.

### LEAF Datasets

**Accuracy and Convergence Comparison** Considering the limited computational capabilities of edge devices, each client's local epoch is set to 1.

![20200219125820.png](http://cdn.ereebay.me/hexo/20200219125820.png)

As shown above, all methods within the FedMeta framework converge faster and more stably, achieving clear improvements. MAML and Meta-SGD reach similar convergence speeds and accuracy on the first two datasets; on Sent140, Meta-SGD performs somewhat better.

The table shows the accuracy of the four methods after several communication rounds. First, it can be seen that FedAvg performs far worse than FedMeta, especially on the image classification task. MAML and Meta-SGD achieve the highest accuracy, with gains of 3.23-14.84 percentage points. It is also found that FedAvg (meta) attains higher accuracy than FedAvg in most cases; there are two exceptions, however, when the support fraction is 20%. Unexpectedly, FedAvg (Meta) even shows a slight drop in accuracy. This may be because, after being finetuned on a small amount of data, the model deviates excessively from the global optimum. Second, as we increase the probability p of the support set, the accuracy of both FedAvg (Meta) and FedMeta improves in almost all cases, but the growth rate of FedAvg (meta) is greater than that of FedMeta. This indicates that FedMeta generalizes better and adapts more effectively to clients with limited data.

**Fairness Comparison** The paper also compares FedAvg and FedMeta across multiple runs, using the distribution of final accuracies. The last row of the previous figure shows kernel density estimates for the different methods. For FEMNIST, MAML and Meta-SGD not only yield a higher mean but also lower variance. For the Shakespeare dataset, FedMeta has larger variance but also a larger mean. For Sent140, the accuracy distributions are roughly the same; still, it can be seen that MAML and Meta-SGD have more clients close to 100%. Overall, for image classification tasks, FedMeta's accuracy distribution is more stable and concentrated.

## Conclusion

The FedMeta framework outperforms the original FedAvg in accuracy, convergence speed, and communication cost.
