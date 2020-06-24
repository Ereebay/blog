---
title: 【西瓜书】2. 模型评估与选择
tags: 西瓜书
mathjax: true
categories: notes
abbrlink: 2109
date: 2020-06-24 16:06:16
---
# 模型评估与选择

## 经验误差与过拟合

错误率（error rate）：分类错误的总占比

精度（accuracy）：分类正确的占比

误差（error）：实际输出与真实输出的差异，在训练样本上为经验误差或训练误差，在新样本上的为泛化误差

过拟合（overfitting）：泛化性能下降

欠拟合（underfitting）：对训练样本效果不好

## 评估方法

通常使用测试集合（testing set）测试模型的性能，以测试误差（testing error）作为泛化误差近似

### 留出法

将数据集D分为两个互斥的集合，一部分为训练集S，一部分为测试集T。当S较大T较小的时候可能评估不够准确。如果T变大了，那么用S训练的模型可能和用D训练的模型相比差别太大，评估的结果也不够准确。

### 交叉验证法

将数据集D分为k个互斥的集合，将k-1个子集作为训练集，剩下的集合做为测试集合，这样就可以获得k组数据，最终返回这k组数据测试结果的平均值。极端情况就是留一法，留一法结果更为准确，但是计算开销大。

### 自助法

采用自助采样法（bootstrapping）有放回的采样一个新的数据集$D'$。用这个新数据集做训练，剩下的做测试。优点：该方法在数据集较小，难以有效的划分训练/测试集的时候有用。缺点：改变初始数据集分布，引入估计偏差。

### 调参与最终模型

算法参数（parameter）对算法的性能有很大影响。除了对算法进行选择，还需要对参数进行调整，就是调参（parameter tuning）。其中在对模型利用验证集对模型评估选择之后，确定模型和参数配置后，需要用数据集D整体重新训练一次模型才能进行测试，这才是最后提交的模型。模型评估时用的数据为了区分，称为验证集（validation set）。用验证集来评估算法的选择和调参，用测试集来评估算法的泛化性能。

## 性能度量

性能度量（perfomance measure）：衡量模型泛化能力的评价标准

回归任务中最常用的性能度量是mse 均方误差：

$$E(f ; D)=\frac{1}{m} \sum_{i=1}^{m}\left(f\left(x_{i}\right)-y_{i}\right)^{2}$$

更一般的可以写成：

$$E(f ; D)=\int_{x \sim D}(f(x)-y)^{2} p(x) d x$$

### 错误率与精度

最常用的两种性能度量：错误率和精度

错误率：$E(f ; D)=\frac{1}{m} \sum_{i=1}^{m} I\left(f\left(x_{i}\right) \neq y_{i}\right)$

精度：$\begin{aligned} \operatorname{acc}(f ; D) &=\frac{1}{m} \sum_{i=1}^{m} \mathbb{I}\left(f\left(x_{i}\right)=y_{i}\right) \\ &=1-E(f ; D) \end{aligned}$

### 查准率、查全率与F1

在信息检索领域，通常关心“检索的信息有多少用户感兴趣”，“用户感兴趣的有多少被检索”。因此有查准率（precision）和查全率（recall）的概念。

二分类问题，

查准率 = 真正例/（真正例+假正例）

查全率 = 真正例/（真正例+假反例）

这两种比例通常是互矛盾，当查全率高，查准率相对就低，比如为了选出尽可能多的好瓜，只要选中所有瓜那必然都选上了。

P-R曲线：查准率-查全率曲线，若一个learner的PR曲线可以包住另一个learner的曲线，则说明该learner 更好。比较合理的判断依据是曲线下面积的大小，但是不容易估算，于是有三种评价手段。

1. 平衡点（BEP）：也就是查准率=查全率的取值。
2. F1度量：F1=2*P*R/(P+R)= 2*TP/（样例总数+TP-TN） 基于P和R的调和平均$\frac{1}{F_{1}}=\frac{1}{2}\left(\frac{1}{P}+\frac{1}{R}\right)$
3. F1度量的一般形式（对P和R的重视程度不同）：$F_{\beta}=\frac{\left(1+\beta^{2}\right) \times P \times R}{\left(\beta^{2} \times P\right)+R}$ 基于调和平均$\frac{1}{F_{\beta}}=\frac{1}{1+\beta^{2}}\left(\frac{1}{P}+\frac{\beta^{2}}{R}\right)$  $\beta$代表了两者的相对重要性，1退化为F1，大于1说明查全率更重要，小于1查准率更为重要。

有的时候需要在n个二分类混淆矩阵上计算P和R，然后计算平均值，这就是宏查准率，宏查全率，宏F1。也可以将混淆矩阵中的对应元素进行平均，得到各个正例反例平均，然后计算微查准率微查全率和微F1。

### ROC与AUC

ROC：受试者工作特征（Receiver Operating Characteristic）曲线。

对于学习器来说通常对输入样本输出一个概率预测，然后与分类阈值（threshold）进行比较。模型将最可能是正例的样本排在最前面，最不可能排在最后面，然后在某个截断点（cut point）将样本分为两部分，前一部分判断为正样本，后一部分判断为负样本。

不同的分类任务，会采用不同的截断点，如果更注重查准率，截断点靠前，如果更重视查全率，截断点靠后。

根据学习器的预测结果对样本进行排序，按此顺序计算两个值。

纵坐标：真正例率（True Positive Rate TPR）：$T P R=\frac{T P}{T P+F N}$

横坐标：假正例率（False Positive Rate FPR）：$FPR=\frac {FP} {TN+FP}$
![Untitled](http://cdn.ereebay.me/hexo/Untitled.png)

学习器比较时，若一个能够包住另一个，则说明前者优于后者，若有交叉，则分秦光，比较合理的判断是比较ROC曲线下面积，即AUC。

$$A U C=\frac{1}{2} \sum_{i=1}^{m-1}\left(x_{i+1}-x_{i}\right)\left(y_{i}+y_{i+1}\right)$$

损失loss：$\operatorname{l_{rank}}=\frac{1}{m^{+} m^{-}} \sum_{x^{+} \in D^{+}} \sum_{x \in D^{-}}\left(\mathbb{I}\left(f\left(x^{+}\right)<f\left(x^{-}\right)\right)+\frac{1}{2} \mathbb{I}\left(f\left(x^{+}\right)=f\left(x^{-}\right)\right)\right)$

AUC= 1-lrank

[https://datawhalechina.github.io/pumpkin-book/#/chapter2/chapter2](https://datawhalechina.github.io/pumpkin-book/#/chapter2/chapter2) 更详细可以看南瓜书

### 代价敏感错误率与代价曲线

为权衡不同类型错误所造成的不同损失，可为错误赋予“非均等代价”（unequal cost）

以二分类为例，设定一个代价矩阵 cost matrix

二分类代价矩阵
|  真实类别| 预测类别0 |预测类别1|
|  ----  | ---|---|
| 0  | 0|cost01|
| 1 | cost10| 0

如果说认为把0判成1的损失更大则cost01大于cost10，相差越大，值的差别越大。

前面介绍的性能度量方式都隐含了均等代价，在非均等代价下，希望总体最小化总体代价 total loss

代价敏感 cost sensitive 错误率为：

$$
E(f ; D ; \cos t)=\frac{1}{m}\left(\sum_{x_{i} \in D^{+}} \mathbb{I}\left(f\left(x_{i}\right) \neq y_{i}\right) \times cost_{01}, \quad+\sum_{X_{i} \in D^{-}} \mathbb{I}\left(f\left(x_{i}\right) \neq y_{i}\right) \times cost_{1 0}\right)
$$

在非均等代价下，ROC曲线不能反映出学习器的期望总体代价，但是代价曲线 cost curve 可以达到目的。

横轴是[0,1]的正例概率代价：

$$
P(+) cost=\frac{p \times cost_{01}}{p \times cost_{01}+(1-p) \times cost_{10}}
$$

p是样例为正例的概率

纵轴是[0,1]的归一化代价：

$$
\cos t_{\text {norm }}=\frac{ F N R \times p \times cost_{01}+FP R \times(1-p) \times \cos t_{10}}{p \times cost_{01} +(1-p) \times cost_{10}}
$$

FNR为假反例率，FPR为假正利率。 FNR=1-TPR

## 比较检验

机器学习中的性能比较通常采用统计假设检验（hypothesis test）。原因如下

1. 比较泛化性能，通过试验评估的是测试集的性能，两者对比结果未必相同
2. 测试集上的性能和测试集的选择有很大关系
3. 算法本身具有一定的随机性

### 假设检验

错误率${\epsilon}$表示性能度量。
对于一个错误率为${\epsilon}$的学习器，我们在测试集上得到测试错误率为$\hat{\epsilon}$ ，这就意味着在$m$个样本中，有$\hat{\epsilon} \times m$个被分类错误，我们容易推出，一个泛化错误率为${\epsilon}$的学习器在一个$m$样本的测试集上，得到测试错误率$\hat{\epsilon}$的概率为：

$$
P(\hat{\epsilon} ; \epsilon)=\left(\begin{array}{c}m \\ \hat{\epsilon} \times m\end{array}\right) \epsilon^{\hat{\epsilon} \times m}(1-\epsilon)^{m-\hat{\epsilon} \times m}
$$

采用二项分布进行假设检验，这里的假设形如$\epsilon \leq \epsilon_{0}$，那么在$1-\alpha$的概率内我们能观测到的最大错误率为：

$$
\bar{\epsilon}=\max \epsilon \quad \text { s.t. } \sum_{i=\epsilon_{0} \times m+1}^{m}\left(\begin{array}{c}m \\ i\end{array}\right) \epsilon^{i}(1-\epsilon)^{m-i}<\alpha
$$

如果测试错误率$\hat{\epsilon}$小于临界值$\bar{\epsilon}$，那么我们就在$1-\alpha$的置信度下接受该假设，否则就以$\alpha$的显著度认为泛化错误率大于$\epsilon$。

如果采用多次留出法或者交叉验证法，则获得k个测试错误率。这个时候可以计算平均值和方差，从而利用t检验进行假设检验。这个时候的假设就是均值为错误率。变量：

$$
\tau_{t}=\frac{\sqrt{k}\left(\mu-\epsilon_{0}\right)}{\epsilon}
$$

服从自由度为k-1的t分布。

更多检验方法在西瓜书p41有说明

## 偏差与方差

除了估计泛化性能，还希望了解为什么有这样的性能，可以使用偏差方差分解。以回归任务为例，yd是x在数据集的标记，y为x的真实标记，f(x,D)是预测的输出。

那么期望预测为：$\bar{f}(x)=\mathbb{E}_{D}[f(x ; D)]$

使用样本数不同的训练集可以产生方差。然后进行分解：

$$
\begin{aligned} E(f ; D) &=\mathbb{E}_{-} D\left[\left(f(x ; D)-y_{D}\right)^{2}\right] \\ &=\mathbb{E}_{-} D\left[(f(x ; D)-\bar{f}(x))^{2}\right]+(\bar{f}(x)-y)^{2}+\mathbb{E}_{-} D\left[\left(y_{D}-y\right)^{2}\right] \\ &=\operatorname{bias}^{2}(x)+\operatorname{var}(x)+\varepsilon^{2} \end{aligned}
$$

泛化误差可以分解为偏差、方差和噪声之和

- 偏差度量了算法的期望预测与真实结果的偏离程度，刻画了学习算法本身的你和能力
- 方差度量了同样大小的训练集的变动所导致的学习性能的变化，刻画了数据扰动所造成的影响
- 噪声表达了当前任务上任何学习算法所能达到的期望泛化误差的下界，刻画了学习问题本身的难度。


