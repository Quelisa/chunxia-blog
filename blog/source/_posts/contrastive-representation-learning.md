---
title: 浅谈对比表示学习
date: 2022-05-11 16:41:52
tags:
- Representation Learning
categories: 
- 对比学习
mathjax: true
---


最近用SimCSE[1]的对比方式对跨语言模型进行再次训练，在文本相似度的任务上效果都有提升，不仅效果好，而且做法还十分简单，非常适合无监督训练。因此打算了解一下对比表示学习。

<!--more-->


### 对比表示学习和sentence embedding
以NLP为例，sentence embedding的核心思想是，语义相似的文本在空间的向量表示彼此相近，而语义无关的文本在空间的向量表示彼此疏远。而对比表示学习正是捕捉到了这种表示学习简单的建模方式，采用一定的数学手段，将样本中的正例也就是语义相近的文本和负例区分来开。所以这里涉及到的核心主要是两个，一个是如何构造正例和负例，另一个就是如何设计度量距离和loss。


### 正例负例设计

一般在对比学习中，正例的设计使用数据增强的方式生成带有噪声的正例，负例采用同一个batch中的其他的数据，所以负例的大小取决于batch的大小，而负例的选择也在一定程度上影响着对比表示学习的效果。同样，正例的数据增强方式也会对表示效果有不同的表现。下面介绍几种经典的数据增强方式：


1. 词汇编辑（同义词替换、随机插入、随机交换、随即删除）
2. Dropout（SimCSE采用此方法，效果较好）
3. Cutoff（token、feature、span）[2]
4. Back-translation



### 度量距离和对比loss
常用的度量距离有欧几里得距离、余弦相似度、马氏距离等，下面按照对比loss的发展介绍几种经典的loss。

#### Contrastive Loss[3]
该论文中采用欧几里得距离作为度量，损失函数定义如下：

$$ \mathcal{L}(W) = \sum_{i=1}^P{L(W,Y,\vec{X_1},\vec{X_2})^i} \qquad (1) $$
$$L(W,(Y,\vec{X_1}, \vec{X_2^2})) = (1-Y)L_S(D^i_W) + YL_D(D^i_W) \qquad (2) $$

其中$W$是模型参数，$X_1$、$X_2$是样本在空间的向量表示，$Y$是0-1变量，取0时表示两个样本一致，取1表示不一致。$L_S$表示正正样本对loss，$L_D$表示正负样本对loss。其中正正、正负loss的设计思路如下图，最后的loss表达式如（3）。

<img src="https://github.com/Quelisa/picture/raw/main/Loss/constrastive_loss.png" width="50%" height="50%">


$$L(W,(Y,\vec{X_1}, \vec{X_2^2})) = (1-Y)\frac{1}{2}(D_W)^2 + Y\frac{1}{2}\{max(0,  m- D_W)\}^2 \qquad (3) $$

可以看到这里设置了一个类似于分类中的margin，当两个样本为同一个类别时，loss变为优化$L_S$，这时候直接最小化他们的欧几里得距离，当两个样本不是同一个类时，loss变为优化$L_D$，这是如果样本距离已经大于margin距离则不对他们进行处理，如果样本距离小于margin距离，则最大化他们的欧几里得距离到margin的距离。


#### NEC（Noise Contrastive Estimation）Loss[5]
为了解决无法直接计算归一化因子的情况NEC，也就是 Noise Contrastive Estimation（噪声对比估计）最初在[4]中被提出，NCE通过对比真实样本与噪声样本，从而能够去估算出真实样本的概率分布的参数。具体做法是讲问题转化为一个二分类问题，在一批含噪声的样本中，判断样本是真实样本还是噪声样本。假设变量真实样本x服从概率$p(x)$，$$p(x) = \frac{e^{G(x)}}{Z}$$, $$Z=\sum_x{e^{G(x)}}$$则称为归一化函数。NEC通过计算$P(1|x)$来绕开直接计算$p(x)$带来的无法计算归一化函数的问题，
$$ p(1|x)=\gamma (G(x;\theta) - \gamma) = \frac{1}{1+e^{-G(x;\theta)+\gamma}} \qquad (4) $$

设$$\tilde{p(x)}$$为真实样本的分布，$U(x)$是服从某一个确定的分布，则其loss为：
$$ \mathcal{L}(\theta,\gamma) = \underset {\theta,\gamma}{arg min} - \mathbb{E}_{x~\tilde{~}{p(x)}}logp(1|x) - \mathbb{E}_{x~\tilde{~}{U(x)}}logp(0|x)\qquad (5) $$
通过公式推导可以得到：

$$ \tilde{p(x)} = exp{G(x;\theta)-(\gamma - logU(x))} \qquad (6)$$

则式中的$$\gamma - logU(x)$$就是归一化常数，因此达到原来的目的。NEC很大的一个贡献是证明了仅适用负采样技术就可以加速归一化项的计算，因此为对比学习提供了很强大的理论支撑。


#### Triplet Loss[6]
正如loss的名字所示，我们将被优化的目标设置为一个三元组$(x,x^+,x^-)$，那么三元组的整体距离为：
$$ \mathcal{L} = min \{d(x,x^+)-d(x,x^-)+\alpha, 0\} \qquad (7)$$

从公式中可以很直观的看到，它通过在使正例到锚点的距离尽可能小，负例到锚点的距离尽可能大的同时还做了一定的约束来防止过拟合。


#### InfoNCE Loss[7]
InfoNCE是引入了互信息的NEC，首先回顾一下什么是互信息。设离散随机变量$X$和$Y$，$p(x,y)$是$X$和$Y$的联合分布，$p(x)$和$p(y)$分别是$X$和$Y$的边缘概率分布函数，则$X$和$Y$的互信息可以定义为：

$$I(X;Y) =\sum_{y\in Y}\sum_{x\in X}p(x,y)log(\frac{p(x,y)}{p(x)p(y)}) \qquad (8)$$


现在假设有一个生成模型$$p(x_{t+k}|c_t)$$根据当前上下文$$c_t$$预测$$k$$个时刻后的数据$$x_{t+k}$$，引入互信息后可以通过最大化当前上下文$$c_t$$未来的数据$$x_{t+k}$$之间的互信息来进行预测，根据条件概率:


$$P(X=a|Y=b) = \frac{P(X=a,Y=b)}{p(Y=b)} \qquad (9)$$


结合公式（8）其互信息表示：
$$I(x_{t+k};c_t) =\sum_{x,c}p(x_{t+k},c_t)log\frac{p(x_{t+k}|c_t)}{p(x_{t+k})} \qquad (10)$$


由于不知道$$c_t$$和$$x_{t+k}$$的联合概率分布$$p(x_{t+k},c_t)$$，所以最大化互信息就要最大化$$\frac{p(x_{t+k}|c_t)}{p(x_{t+k})}$$。根据NCE中提供的思路，将问题转换为一个二分类的问题，从条件$$p(x_{t+k}|c_t)$$中取出的数据为正样本，将它和上下文一起组成正样本对，类标签是1，将$$p(x_{t+k})$$取出的样本为负样本对，它是和当前上下文没有必然联系的随机数据，将它和上下文构成负样本对，类标签为0。根据NCE中说明的设定，正样本选取1个；因为在NCE中证明了噪声分布与数据分布越接近越好，所以负样本就直接在当前序列中随机选取，负样本数量越多越好。所以最后将根据$$c_t$$预测$$x_{t+k}$$的问题就转化为区分正负样本的能力，其对应的交叉熵损失即InfoNCE loss为:


$$  \mathcal{L}_{N} = -\sum_{X}[p(x,c)log \frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in X}f_k(x_j,c_t)}] = -\mathbb{E_x}[log\frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in X}f_k(x_j,c_t)}]\qquad (11) $$

实际上，InfoNCE不仅可以作为自监督学习中的对比损失函数，还可以作为互信息的一个估计器。

#### SCL Loss[8]
SCL loss即监督对比学习损失，是2021年斯坦福提出的在fine-tune阶段对交叉熵loss的改进损失。一般认为交叉熵loss具有泛化性能差和稳定性不好的缺点，尤其是在少量标签的样本中fine-tune时，介于此并考虑到对比学习可以利用少量的样本进行有效的学习，因此提出监督对比损失。监督对比损失没有像其他的对学习方法中那样采用自监督的学习方式，通过数据增强制造正样本，而是采用了监督任务的对比目标。训练目标的总loss是交叉熵loss和SCL loss的加权和：

$$ \mathcal{L} = (1-\lambda)\mathcal{L}_{CE} + \lambda \mathcal{L}_{SCL} \qquad (12) $$
$$ \mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^N\sum_{c=1}^C y_{i,c} \cdot log \hat{y}_{i,c} \qquad (13) $$


$$ \mathcal{L}_{SCL} = \sum_{i=1}^N -\frac{1}{N_{y_i} - 1} \sum_{j=1}^N\mathrm{l}_{i \not= j}\mathrm{l}_{y_i \not= y_j} log\frac{exp(\Phi(x_i)\cdot \Phi(x_j)/\tau)}{\sum_{k=1}^N\mathrm{l}_{i \not= k}exp(\Phi(x_i)\cdot \Phi(x_k)/\tau)} \qquad (14) $$


### 参考文献

[1] Gao T, Yao X, Chen D. Simcse: Simple contrastive learning of sentence embeddings[J]. arXiv preprint arXiv:2104.08821, 2021.
[2] Shen D, Zheng M, Shen Y, et al. A simple but tough-to-beat data augmentation approach for natural language understanding and generation[J]. arXiv preprint arXiv:2009.13818, 2020.
[3] Hadsell R, Chopra S, LeCun Y. Dimensionality reduction by learning an invariant mapping[C]//2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06). IEEE, 2006, 2: 1735-1742.
[4] Gutmann M, Hyvärinen A. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models[C]//Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010: 297-304.
[5] Mnih A, Teh Y W. A fast and simple algorithm for training neural probabilistic language models[J]. arXiv preprint arXiv:1206.6426, 2012.
[6] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 815-823.
[7] Van den Oord A, Li Y, Vinyals O. Representation learning with contrastive predictive coding[J]. arXiv e-prints, 2018: arXiv: 1807.03748.
[8] Gunel B, Du J, Conneau A, et al. Supervised contrastive learning for pre-trained language model fine-tuning[J]. arXiv preprint arXiv:2011.01403, 2020.
