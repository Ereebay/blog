---
title: Federated Meta-Learning with Fast Convergence and Efficient Communication
categories: notes
mathjax: true
abbrlink: 35664
date: 2020-02-05 18:36:17
tags:
---

# Federated Meta-Learning with Fast Convergence and Efficient Communication

## Abstract

本文提出了一个联邦元学习框架FedMeta，该结构共享参数化算法（meta learner）而不是以前的全局模型。本文在LEAF数据集和实际的数据集进行了评估，并且证明了FedMeta所需的通信成本降低了2.82-4.33倍，收敛速度更快，与FedAVG相比甚至提高了3.23~14.84个百分比。此外FedMeta保留了用户隐私，因为仅仅分享了参数化算法而没有数据。

## Introduction

在联邦学习领域比较著名的FedAvg算法，可以灵活地使用SGD进行本地训练，在平衡计算和通信成本的情况下达到较高的精度。

在元学习领域，像MAML这种基于初始化的元学习算法非常擅长对于新的task的快速收敛以及拥有良好的泛化性。这使得其非常适合于non IID且高度个性化的去中心化数据。

受到这个启发，本文开发了一个与之前联合学习中工作大不相同的联合元学习框架。本文首先将元学习方法和联邦学习联系起来。在元学习中，通过元训练过程从大量的任务重缓慢学习参数化的算法（meta learner），在该过程中，算法会在每个task中快速训练一个特定的模型。

task通常由support set和query set组成，彼此不想交。在support set上训练特定任务的模型，在query set上进行测试。然后利用测试结果更新算法。在联邦元学习中，算法（meta learner）在server端维护，并分发到clients进行模型的训练。元训练的每个episode过程中，被采样到的一个batch的clients会接受到算法（meta learner）的参数然后进行模型训练。然后将query set上的测试结果上传到server端更新 meta learner。整体流程如图所示：

![20200218221422.png](http://cdn.ereebay.me/hexo/20200218221422.png)

**Comparing federated meta-learning with federated learning** FML和ML的比较。
联邦元学习类似于联邦学习，区别主要在于共享的不再是全局模型，而是算法参数（元模型参数）。但是，元学习在概念上和分布式模型训练不同，而且在共享元学习算法可以比共享模型更灵活应用。例如在图像分类中，n个类别的图像可能在clients之间不均匀地分布，其中每个client最多具有k个类别。对于联邦学习需要训练一个大型的n类分类器，以利用来自所有客户端的数据，然而其实k类分类器就足够了，因为他每次其实只为一个client做预测。对于联邦学习这种庞大的模型需要大量的通信成本，虽然可以只向client发送模型的一部分以更新相关参数，但这必须先了解client的私有数据才能决定。而另一方面，在元学习中，算法可以训练包含不同类别的任务。例如MAML可以通过对k类任务进行元训练来为K类分类器提供初始化，无论具体类别是什么。因此，在FML框架中，可以利用MAML对所有n个类别进行k类分类器初始化的元训练。也就是让一个k类分类器在n个类别的任务背景下进行初始化的元学习。这样FML的通信和计算成本都降低了。

**Contributions** 本文贡献主要专注于联邦学习设置方面的算法设计，为此本文提出了一个新的框架并做了大量的实验。贡献点主要有三：1. 提出FedMeta框架，结合meta和fed，将maml算法和meta-sgd集成到federated learnin中。2. 在LEAF数据集上进行试验，与FedAvg进行比较，结果表明Fedmeta减少开销，同时精度更高。3. 将FedMeta应用于推荐任务，其中每个client都有高度personalized的记录，通过实验证明，与独立或联邦学习方法系那个比，元学习算法可以实现更高准确性。

## Federated Meta-Learning

### The Meta-Learning Approach

元学习的目的是meta-train一个算法（meta learner）A，能够快速的训练一个模型。也就是能够得到一个能够快速收敛的初始模型。算法$A_{\varphi}$是参数化的模型，其参数在元训练过程中，通过一系列的tasks进行更新。
元训练过程中，一个task T包含了一系列的 support set和query set，每个set都有对应的标签和数据。算法（meta learner）A会在support set上对模型f进行训练，并且输出参数$\theta_T$，这成为inner update。然后模型$f_{\theta_T}$会在query set上进行评估，然后会计算test loss$\mathcal{L}_{D_{Q}^{T}}\left(\theta_{T}\right)$ 来反映算法A的训练能力。最后A通过最小化test loss来进行更新，这个步骤称为outer update。这里要注意的是query set和support set是disjoint的，以最大化A的泛化能力。元训练阶段中在每个episode会从一个meta training set中样一个batch的tasks。因此算法A的优化目标可以表示为：

$$
\min _{\varphi} \mathbb{E}_{T \sim \mathcal{T}}\left[\mathcal{L}_{D_{Q}^{T}}\left(\theta_{T}\right)\right]=\min _{\varphi} \mathbb{E}_{T \sim \mathcal{T}}\left[\mathcal{L}_{D_{Q}^{T}}\left(\mathcal{A}_{\varphi}\left(D_{S}^{T}\right)\right)\right]
$$

MAML是具有代表性的元学习算法。对于MAML的算法A就是用于产生模型的初始状态。具体地说，就是对于每个task T，算法使得$\alpha = \theta$，使得算法的参数和模型f的参数相等。然后模型f的参数在support set上训练，根据损失函数进行更新：$\mathcal{L}_{D_{S}^{T}}(\theta):=\frac{1}{\left|D_{S}^{T}\right|} \sum_{(x, y) \in D_{S}^{T}} \ell\left(f_{\theta}(x), y\right)$。最后把模型参数在query set上进行测试，然后计算测试的损失：$\mathcal{L}_{D_{Q}^{T}}\left(\theta_{T}\right):=\frac{1}{\left|D_{Q}^{T}\right|} \sum_{\left(x^{\prime}, y^{\prime}\right) \in D_{Q}^{T}} \ell\left(f_{\theta_{T}}\left(x^{\prime}\right), y^{\prime}\right)$

公式1可以简写为：

$$
\min _{\theta} \mathbb{E}_{T \sim \mathcal{T}}\left[\mathcal{L}_{D_{Q}^{T}}\left(\theta-\alpha \nabla \mathcal{L}_{D_{S}^{T}}(\theta)\right)\right]
$$

基于MAML，Meta-SGD进一步学习初始参数和内循环学习率在同一时刻。测试损失可以看做是模型参数和学习率的函数，两者都可以在外循环中采用SGD算法求梯度来被更新。此外，学习率和模型参数是相同维度的向量，是的学习率在坐标方向上和模型参数向量相对应。

因此优化条件可以重写为 

$$
\min _{\theta, \alpha} \mathbb{E}_{T \sim \mathcal{T}}\left[\mathcal{L}_{D_{Q}^{T}}\left(\theta-\alpha \circ \nabla \mathcal{L}_{D_{S}^{T}}(\theta)\right)\right]
$$

### The Federated Meta-Learning Framework

FML的 目标是能够协作式的通过分布在各个clients的数据meta train一个算法。以MAML为例，就是希望那个利用所有clients的数据，来完成一个模型的初始化。MAML包括两个层次的优化过程：内循环利用初始化参数训练特定任务模型，外循环根据测试损失更新初始化参数。在联邦学习的环境中，每个client u会从服务器接收初始化参数 $\theta$，然后根据设备上的support set数据进行模型的训练，并传输test loss到server。server主要维护初始化参数，并根据clients的test loss对其进行更新。

在这个过程中传输的信息包括了：模型初始化的参数（从server 到clients）和test loss（从clients到server）。对于Meta-SGD算法，向量$\alpha$也是传输的部分，用于内循环模型训练的学习率参数。

![20200219111333.png](http://cdn.ereebay.me/hexo/20200219111333.png)

算法1描述了采用MAML和MetaSGD的FedMeta的过程，其中communication round表示的是episode。算法在AlgorithmUpdate步骤中被维护。在更新的每轮，server会收集MAML或者metasgd在采样的clients上训练的得到的test loss。初始化参数$\theta$在client的训练集训练后更新，然后使用更新后的参数在测试机上测试。然后在meta trian过程后将模型部署在client上。

## Experiments

1. 在LEAF数据集上实验，证明了其收敛性块，准确性高。
2. 在实际环境下的推荐任务中进行实验，可以保证在较小的规模依然保持较强的适应力。

### Evaluation Scheme

所有实验中80%的clients作为训练clients，10%作为验证clients剩余的作为测试clients。每个clients的本地数据会分为support set和query set。同时本文改变每个client的p部分的数据作为support set来研究FedMeta如何有效适应数据量有限的新client。后面简称为psupport。

![20200219114337.png](http://cdn.ereebay.me/hexo/20200219114337.png)

作者做了FedAVG，FedAVG的meta版本以及两种FedMeta的实验。FedAVG的meta版本是利用测试clients的support set 来对从server接收到的初始化模型进行finetune，然后测试。而训练阶段，两者都采用训练clients上所有的数据。

至于FML，测试了三种MAML，FOMAML，Meta-SGD，都是模型无关的方法，而且容易实现。

### LEAF Datasets

**Accuracy and Convergence Comparsion** 考虑到边际设备的运算能力的限制，每个client的local epoch设置为1。

![20200219125820.png](http://cdn.ereebay.me/hexo/20200219125820.png)

如上图所示，FedMeta框架所有方法都更快更稳定的收敛了，并且得到了明显的提升。MAML和Meta-SGD的收敛速度很准确率在前两个数据集都达到了相似的收敛速度和准确度。在Sent140上meta-sgd更好一些。

表格展示了四种方法数轮通信后的准确度。首先可以看出，FedAvg表现远远差于FedMeta，尤其是图像分类的任务。MAML和Meta-SGD的准确度最高，增加了3.23-14.84的准确度。同时也发现，fedavg（meta）在大多数情况比fedavg准确率更高。然而，有两个例外，就是当support fraction为20%的时候。出乎意料的是FedAvg（Meta）还有在精准度上轻微的下降。这可能是因为在对少量数据进行微调后，该模型和全局最优值存在过度偏差。其次，当我们增加support集的概率p，几乎在所有情况下FedAvg（Meta）和FedMeta的精度都会提高。但是FedAvg（meta）的增长率大于FedMeta。说明FedMeta泛化能力更强，更有效适应数据量有限的client。

**Fairness Comparison** 本文同时还比较了多次实验，利用平均最终的准确度分布，来研究FedAvg和FedMeta。从前面的图的最后一行，展示了不同方法的核密度估计。对于FEMNIST，MAML和Meta-SGD不仅导致较高的均值，而且方差更低。对于Shakespeare数据集，FedMeta方差虽然更大，但是均值也更大。对于Sent140，准确度的分布都差不多。然而可以看出，MAML和Meta-SGD有更多接近100%的clients。总之，对于图像分类任务，FedMeta的准确度分布更为稳定集中。

## Conclusion

FedMeta框架，在准确性 收敛速度和通信成本都比原本的FedAvg要更好。
