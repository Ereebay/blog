---
title: "Stagewise Knowledge Distillation"
date: 2020-02-17T13:15:34+08:00
draft: false
math: true
categories: [notes]
---

# Stagewise Knowledge Distillation

## Abstract

Most modern deep learning models require high computational power, but embedded devices lack such computing capability. Therefore, for this kind of device, models that can reduce computation while maintaining performance are very important. Knowledge distillation is one of the methods that solves this kind of problem. Traditional knowledge distillation methods directly transfer knowledge from the teacher to the student in a single stage. We propose a stagewise training approach to improve knowledge transfer. This approach can even use only a portion of the data used to train the teacher model without affecting performance. This approach can complement other model compression techniques, and can even be regarded as a universal model compression technique.

<!-- more -->

## Introduction

This paper is mainly about model compression techniques based on knowledge distillation, so let me first introduce the categories of model compression.

Model compression can mainly be divided into the following five categories:

1. Parameter Pruning and Sharing: mainly aims to reduce redundancy in network parameters and eliminate unnecessary parameters.
2. Low Rank Factorization techniques: mainly use tensor/matrix factorization to determine the effective parameters of the network.
3. Transferred/Compact Convolutional Filters: aim to use specially designed convolutional filters to reduce computation and storage space.
4. Knowledge Distillation: aims to use a larger pretrained model, the teacher, to train a small model, the student.
5. Quantization: aims to reduce the number of bits of each weight while preserving network performance.

This work focuses mainly on the knowledge distillation approach. Ideally, the teacher should be able to pass on all the knowledge it has learned to the student, but this is not the case. Moreover, not all of the teacher's knowledge is necessarily relevant to the student; the ideal situation is that the student learns the important parts and ignores the unimportant ones. This paper mainly uses ResNet34 as the teacher model, and the student model also uses a ResNet-like model, but is much smaller in storage structure and computational complexity. This paper presents a method of training the student using multiple feature maps of a pretrained teacher model.

This paper mainly adopts a new way of training: the student model is trained using the teacher model's feature maps fixed at a particular layer. For each feature map, the student model is trained in a stagewise manner, and the final classification layer is trained directly on the dataset without the teacher. It is finally demonstrated that this approach can learn directly on a subset of the teacher's training data.

## Related Work

## Methodology

This paper mainly uses ResNet networks; the specific structure of ResNet will not be repeated here. It mainly consists of:

- Basic Block
- Downsample Block
- ResNet18 or 34 type models

### Teacher Network

This paper uses ResNet34 as the teacher model.

### Student Network

A reduced version of ResNet34 is used as the student model, mainly by reducing the number of basic blocks.

### Dataset

Three datasets are used: Imagenette, Imagewoof and CIFAR10. The first two datasets are subsets of ImageNet. The first is relatively easy, and the latter is relatively harder. The purpose of this work is not to improve accuracy as much as possible, but to make the student's accuracy as close as possible to the teacher's.

### Proposed Training Method

In the early stage of the experiments, we trained the student model's multiple feature maps to simultaneously mimic the corresponding feature maps of the teacher model and the labels. So the mean squared error of each pair of feature maps would be accumulated. In addition, the cross-entropy loss is also accumulated. Therefore the total loss function can be expressed as:

$$
\begin{aligned} L\left(y, \hat{y}, y_{c l s}, \mathrm{class}\right) &=\frac{1}{M N} \sum_{i=1}^{N} \sum_{j=1}^{M}(y(i, j)-\hat{y}(i, j))^{2} \\ &+\frac{1}{M C} \sum_{j=1}^{M} \sum_{k=1}^{C}\left\{-\log \left(\frac{\exp \left(y_{c l s}(j, \text { class })\right)}{\sum_{k=1}^{C} \exp \left(y_{c l s}(j, k)\right)}\right)\right\} \end{aligned}
$$

N denotes the number of blocks, and y(i,j) is the intermediate output of the teacher model's i-th block for the j-th input. Likewise, y-hat is that of the student model. M is the batch size. y_cls(j,k) is the model's output for the j-th input with respect to the k-th class, C is the number of classes, and class is the correct class that each particular input represents.

These early experiments showed a very small improvement of student models trained with the teacher model over those trained without. This can be attributed to the fact that multiple feature maps and labels must be mimicked at the same time, i.e., the conditions imposed on the optimization algorithm are very strict. Assigning weights to each MSE loss and the cross-entropy loss did not help either, because the training process was still very strict. Another possible reason is gradient vanishing and accumulation. To reduce this strictness of training, a stagewise training approach was proposed.

We train the student model in a stagewise manner, i.e., one block at a time. The image is the input to both the teacher and the student model, and the outputs of the first block are taken from both models. The MSE error is applied between the outputs, and then backpropagation is performed on the student model. After training the first block for 100 epochs, training stops. In the next step, the input is again passed to the teacher and the student, but the features of the second block are taken, and the same procedure as in the first stage is followed, i.e., the MSE loss between the outputs of the second block. Then backpropagation for 100 epochs. This operation is repeated for all the blocks. At the end of the student model, the classifier part is trained directly to predict classes from the dataset, i.e., the images are passed through the student model and trained with the cross-entropy loss for class prediction. At this stage, no teacher model is used, and the rest of the student model (the parts of the model other than the classifier part) is frozen. Figure 2 can help understand this point. The training loss function of stage i can be expressed as:

$$
L_{i}(y, \hat{y})=\frac{1}{M} \sum_{j=1}^{M}(y(i, j)-\hat{y}(i, j))^{2}
$$

The classifier uses the standard cross-entropy loss:

$$
L_{c l s}\left(y_{c l s}, \mathrm{class}\right)=\frac{1}{M C} \sum_{j=1}^{M} \sum_{k=1}^{C}\left\{-\log \left(\frac{\exp \left(y_{c l s}(j, \mathrm{class}\right)}{\sum_{k=1}^{C} \exp \left(y_{c l s}(j, k)\right)}\right)\right\}
$$

We have shown that stagewise training has its own advantages; the main advantage is that the number of parameters to be optimized at a time is limited. Compared with training a larger number of parameters at once, this limited number of parameters can relax the strictness during training. The results show that stagewise training works better than training everything at once.

#### Less Data Approach

Datasets like ImageNet are so large that performing stagewise training of the student model with the teacher model on limited hardware would take a great deal of time. Therefore, it becomes useful if we can perform stagewise training using only a subset of the data while maintaining accuracy. Thus, the stagewise training experiments were repeated using 1/4 of the original training data. Note that the original training data refers to the data on which the teacher model was trained, and the remaining 3/4 of the data is kept as a test set for evaluation.

## Results

Figures 3 and 4 give the results of simultaneous training, stagewise training, and stagewise training with partial data.

These figures show that the student model with the entire dataset achieves almost the same accuracy as the teacher. But in the case of less data, there is a huge gap in accuracy between training with and without a teacher. The following paragraphs discuss the possible reasons behind these results. It should be pointed out that the purpose of model compression is to reduce the gap between teacher and student, not to obtain better accuracy. Obviously, if a better teacher is used, the student's accuracy will improve; sometimes the student's accuracy even exceeds that of the teacher model.

The experimental results can be explained as follows. Since the teacher has already learned the complete dataset, it has already learned the features necessary for classifying the entire dataset. When this teacher is used to train the student, even if training uses a small dataset, its "knowledge" is passed on to the student. Using less data can also be justified by the number of parameters that must be trained in a single stage. Since the proposed method trains only a small part of the network at a time, the number of parameters to be optimized is much smaller than that of the full network mentioned in the preceding paragraphs. The results show that this approach greatly improves accuracy: without a teacher, the accuracy of the student network trained on a small dataset is much lower than that of the student trained on the same dataset with the proposed method. Of course, the main advantage is the reduction in training time, which is very important, because without the teacher, stagewise training would take N+1 times as long. (Because each stage trains the same number of epochs separately.) Here N denotes the number of stages.

Figure 4 shows the experimental results using a smaller amount of data. It can be seen that when using less data and training independently, the student performs very poorly. On the other hand, if a teacher trained on less data is used to train the student stagewise, the prediction accuracy improves greatly. This capability would be very useful when applied to very large datasets.

The results of simultaneous training are close to those of training without a teacher. In particular, for the two ImageNet subsets, simultaneous training is slightly better, but for CIFAR10 it is slightly worse. This again demonstrates that the conditions of simultaneous training are too strict, and there is no obvious advantage over training without a teacher. However, the results of stagewise training are much better than those of both simultaneous training and training without a teacher. Since simultaneous training on the complete dataset did not give optimistic results, it was not performed on smaller datasets.

## Conclusion

This work presents a novel method for transferring knowledge from one network to another. Since the number of parameters optimized in one stage is reduced, the method performs better than transferring knowledge directly with the full network all at once. This also allows the student network to be trained with less data than the teacher. It will be very useful when training on larger datasets such as ImageNet.

In addition, the method is very flexible and can be used together with other model compression techniques and with other models. At the same time, it is not limited to image classification and can also be used in applications such as object detection and image segmentation. It can be regarded as a universal compression technique.
