---
title: 关于联邦学习的个性化能力综述
date: 2020-04-29 12:37:55
tags: notes
categories: notes
---

# 关于联邦学习个性化的综述

## Abstract

联邦学习的目的是为了能够获得一个共享的全局模型，供所有节点使用。但是由于Non-IID的数据分布，导致很多时候，有些仅采用本地数据训练的本地模型的表现反而优与全局模型，这使得这些节点不太愿意参与到这个联邦的过程中。本文会介绍一些目前用于对全局模型进行个性化定制来提升在独立节点上的效果的技术。

## Introduction

联邦学习是一种针对Non-IID数据的分布式机器学习算法，它能够在不需要分享各自节点的数据的情况下使得多个节点协同训练一个共享的全局模型。在每一轮的训练开始，中心节点服务器会向每个节点传送当前的全局模型，每个节点会将模型在本地节点的数据上进行训练。然后中心节点会从所有的节点中收集模型的更新数据并更新到中心节点的全局模型。

节点参与联邦学习的主要目的是为了能够获得更好的模型。当节点的数据量不充分的时候，无法得到一个比较好的本地模型时，就能够通过联邦学习获得一个不错的模型。但是对于那些拥有充分数据的节点来说，联邦学习的好处还存在疑问。Yu在Salvaging Federated Learning by Local Adaptation文中提到，对于许多task来说，有些节点因为全局共享模型不如本地数据训练出来的模型而无法获得好处。Hanzely则在Federated Learning of a Mixture of Global and Local Models文中质疑全局模型的和用户的经典使用用途偏离的太远。

本文的目的是调研近期针对于本地节点构建个性化模型的技术，这些个性化模型在本地节点表现需要由于全局共享模型，或者是本地节点所训练出来的模型。

## Need for Personalization

Wu在Personalized federated learning for intelligent iot applications中阐述了联邦学习系统面对个性化问题的主要的三个挑战

1. 设备在存储计算和通信能力方面的异构性。 
2. 由于NonIID分布导致的数据异构性。 
3. 由于不同用户环境导致需要不同模型而产生的模型异构性。

为了解决数据异构和NonIID数据所带来的挑战，对于全局模型global model的个性化变的越来越重要。大部分的个性化技术通常包括两个步骤。1. 利用各节点协作的方式学习一个global全局模型 2. 利用每个节点的隐私数据对global全局模型进行个性化定制。Jiang认为仅仅依靠全局模型的准确度进行优化难以产生个性化较好的模型，并提出，为了能够使得联邦学习的个性化在实际中得到利用，必须同时并不独立的解决以下三个目标。

1. 建立改进的个性化模型，能够使得大多数节点受益
2. 建立更准确的全局模型，对于少量数据的节点可以更好的进行个性化
3. 能够在少量的训练轮次中实现模型的快速收敛

此外，本地模型中的数据可能只有一部分的样本和特定的任务相关，而且的样本都是无关样本，会影响模型的训练。Tuor提出了一种方法，其中使用基于小型benchmark构建的相关性模型来区分相关样本和不相关样本，然后在联邦学习过程中仅使用相关样本。

## Techniques

### Adding User Context

在介绍针对各个节点个性化全局模型的方法之前，先必须指出一点就是共享的全局模型也可以进行高度个性化的预测，前提是如果将节点的context和personal information进行适当地特征化，并整合到数据集当中。但是大多数的公共数据集不包含上下文特征，开发能够有效合并上下文内容的技术仍然是一个重要的开放问题，而这个技术具有能够提升联邦学习模型的巨大潜力。与此同时，是否可以在不对隐私造成不利影响的情况下进行这种背景的特征化还有待研究。作为单个全局模型和纯粹的本地模型的中间方法，Masour在Three approaches for personalization with applications to federated learning建议用户进行聚类，将相似的客户分组在一起，并且为每个组训练一个单独的模型。

### Transfer Learning

迁移学习可以使得深度学习模型利用在解决A问题上所获得的的能力来解决另一个相关问题。Schneider和Vlachos在Mass personalization of deep learning讨论了使用迁移学习来实现在非联邦环境下的模型个性化。迁移学习也同样被应用于联邦环境中。Wang在Federated evalua- tion of on-device personalization提出将训练后的全局模型的某些或者全部参数将根据本地数据进行重新学习。在前面的引用中提供了其理论依据。通过使用经过训练的全局模型的参数来初始化对本地数据的训练，迁移学习能够利用全局模型所提取的知识，而不需要从头开始学习。同时为了避免灾难性以往的问题，必须要注意，不能对本地数据进行太长时间的再训练/finetune。有一种变体的技术是将全局模型的基础网络层进行freeze，仅仅根据本地数据对top层进行训练。（类似冻结feature层，finetune全连接层）

### Multi-task Learning

在多任务学习当中，模型可以通过联合学习来利用任务之间的共性和差异从而同时解决多个相关任务。Smith在Federated multi-task learning文中展示了多任务学习是一种天然的选择去构建个性化的联邦模型，并开发了MOCHA算法。然而在联邦环境中使用多任务学习有一个缺点是，由于它会为每个任务生成一个模型，因此所有节点都必须参与每一轮的训练。

### Meta Learning

元学习涉及到多个学习任务的训练，以生成能够快速适应的模型，该模型可以通过少量的训练样本就能够快速拟合学习解决新任务。Finn提出了模型无关的元学习算法(MAML)，该算法与使用梯度下降法训练的任何模型都兼容。MAML建立了适用于多个任务的内部表示，因此针对新任务，对于顶层的微调可以产生比较好的结果。

Jiang指出可以将联邦学习的过程看做是meta training而personalization过程可以看做是meta testing过程。那么FedAvg算法与Reptile非常相似。同时作者观察到，仔细的微调可以产生准确率高的全局模型，并且比较容易个性化，但是单纯的根据全局模型的准确率来优化模型会损失模型后续的个性化能力。联邦学习的其他个性化方法将全局模型的生成和个性化能力视作两个独立的过程，Jiang提出了一种改进的FedAVG算法，该算法可以同时获得更好的全局模型和更好的个性化模型。

Fallah在Personalized federated learning: A meta-learning approach,文中提出的标准联邦学习问题的新公式结合了MAML，并试图找出一个全局模型，该模型在每个节点针对其自身的损失函数进行更新后均表现良好。此外，他们提出了Per-FedAvg来解决上述问题。Khodak在Adaptive gradient-based meta-learning methods中提出了ARUBA，并通过将其应用于FedAVG证明了性能的提高。chen在Federated meta-learning for recommendation提出了一个用于构建个性化推荐模型的联邦元学习框架，其中算法和模型都已参数化并且需要优化。

### Knowledge Distillation

Caruana已证明可以将一组模型的知识压缩为一个易于部署的单个模型。知识蒸馏进一步发展了这种想法，并通过让学生模仿老师，将大型教师网络的知识提取到较小的学生网络中。过拟合在个性化过程中造成了巨大的阻碍，尤其是对于本地数据集较小的节点。Yu提出通过将全局模型看做是老师，而节点的个性化模型看做是学生，可以缓解过拟合的现象。Li提出了FedMD，一种基于知识蒸馏和迁移学习的联邦学习框架能够让节点独立的设计他们的网络并利用节点的隐私数据和全局的公共数据集。

### Base+Personalization Layers

在传统的联邦学习环境下，数据分布与多个参与训练的设备中。为了缓解这种数据异构性所带来的不利影响，Arivazhagan等人在Federated learning with personalization layers文中提出了FedPer网络，这种网络的基础网络层由FedAVG进行训练得到，而顶层（个性化层）则通过本地节点的变体梯度下降法寻你练得到。迁移学习的方式，是将所有网络层现在全局数据上训练，然后在本地节点数据上，对某些层或者所有层进行再训练。而FedPer算法是分别在全局数据上训练基础网络层，在本地数据上训练个性化层。

### Mixtrue of Global and Local Models

Hanzely在Federated learning of a mixture of global and local models提出了一种新的问题形式，试图在全局模型和局部模型之间寻找平衡。每个节点不学习单个全局模型，而是学习全局模型和他自己的局部模型的混合体。为了解决这个问题，作者提出了一种梯度下降法的变体，LLGD，Loopless Local Gradient Descent(LLGD)。与完全平均不同的是，这种方法仅采取步骤求平均，因为完全平均的方法可能过于激进。

## Discussion

在联邦学习中，当本地节点的数据集很小，且都是IID的情况（是不是和元学习的场景相似）下，全局模型通常会超过本地模型，而且大部分的节点都会受益于联邦学习的过程。然而，当节点拥有充分大量的隐私数据集，并且数据的分布是nonIID的时候，本地模型通常会优于全局模型，而且节点通常不倾向与参与到联邦学习过程中。**一个开放的理论问题就是：如何决定什么时候全局模型的表现会由于单节点上的模型**

这篇文章主要总结了几种用于优化全局模型个性化技术。除了少数的例外，大多数之前的工作都集中在衡量全局模型在聚合的数据上的表现，而不是衡量这些模型在单独节点上的性能。但是如果全局模型会在使用之前进行个性化设置的话，那么全局性能就没有意义。

个性化模型通常在单节点上的表现能够优于全局模型和本地模型。但是在某些情况下，个性化模型的能力无法达到和本地模型相同的能力，尤其是在差分隐私等情况下。