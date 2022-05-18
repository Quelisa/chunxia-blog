---
title: 重温经典Word2vec & GloVe
date: 2021-10-19 16:42:15
tags:
- 静态词向量
- 负采样
categories: 
- 预训练语言模型
mathjax: true
---

Word2vec由Mikolov在2013年提出，GloVe是对Word2vec的改进，它融合了全局统计信息。虽然随着BERT的出现，这种方法已经很少使用，但是其方法仍然值得思考借鉴。

<!--more-->

### 背景
传统的语言模型在进行自然语言任务的时候首先要将词转化为向量，然后输入到模型中进行分类或者标注任务。如何将词转化为向量呢？比较简单的想法是one-hot编码，但是one-hot有一个缺点是数据太多稀疏，而且语义没有连续性。因为语言之间存在天然的上下文联系，这种关系对于单个词的表示有很大的影响，于是研究者开始想能不能用一个模型将所有的词映射到空间中，使得语义相似的词距离较近，而语义无关的距离疏远呢？于是就有了经典现代静态词向量算法Word2vec[1]和GloVe[2]。

### Word2vec
Word2vec基于一个自然语言中有意思的假设：一个词的语义可以用它的上下文表示。因此，word2vec可以用两个模型表示，即连续词袋模型（CBOW）和跳元模型（Skip-gram），模型结构如图：
<img src="https://github.com/Quelisa/picture/raw/main/DNN/word2vec.png" width="60%" height="40%">


下面分别介绍一下这两模型。

#### 连续词袋模型（CBOW）
连续词袋模型假设中⼼词是基于其在⽂本序列中的周围上下⽂词⽣成的。例如，在⽂本序列“the”、“man”、“loves”、“his”、“son”中，在“loves”为中⼼词且上下⽂窗⼝为2的情况下，连续词袋模型考虑基于上下⽂词“the”、“man”、“him”、“son”⽣成中⼼词“loves”的条件概率，即：

$$P(``love"|``the",``man",``his",``son")$$


CBOW模型可以分为输入层、词向量层和输出层。
1. 在输入层中，假设上下文窗口大小为5，则在目标词$w_t$左右各取两个词作为模型的输入。输入层由四个维度为词表长度$\mathbb{V}$的独热表示向量构成。
2. 词向量层将输入层中每个词的独热表示向量经由矩阵$E\in \mathbb{R}^{d \times |\mathbb{V}| }$映射到词向量空间：
$$ \pmb{v}_{w_i}=E \pmb{e}_{w_i} $$
设$$w_i$$对应的词向量即为矩阵$$E$$中相应位置的列向量，$$E$$为由所有向量组成的矩阵。令$$\mathcal{c}_t = \{ w_{t-k}, \cdots, w_{t-1}, w_{t+1},  \cdots, w_{t+k} \}$$表示$$w_t$$的上下文单词集合。对$$\mathcal{c}_t$$中所有词向量取平均，就得到了$$w_t$$的上下文表示：
$$\pmb{v}_{\mathcal{C}_t} = \frac{1}{ |\mathcal{C}_t| } \sum_{w\in \mathcal{C}_t} \pmb{v}_m$$
3. 输出层根据上下文表示对目标词进行预测（分类），令$$E' \in \mathbb{R}^{ |\mathbb{V} \times d | }$$为隐含层到输出层的权值矩阵，记$$\pmb{v}_{w_i}^{'}$$为$$E'$$中与$$w_i$$对应的行向量，那么$$w_t$$的概率可以由下式计算：
$$P(w_t|\mathcal{C}_t) = \frac{exp(\pmb{v}_{\mathcal{C}_t} \cdot \pmb{v}_{w_t}^{'})}{\sum_{w' \in \mathbb{V}}exp(\pmb{v}_{\mathcal{C}_t} \cdot \pmb{v}_{w'}^{'})}$$



在CBOW模型的参数中，矩阵$E$和矩阵$E'$均可作为词向量矩阵，他们分别描述了词表中的词作为条件上下文的或者目标词时的不同性质。



#### 跳元模型（Skip-gram）
跳元模型则适用当前词预测其上下文的词，某种程度上讲Skip-gram模型是CBOW的简化，它建立的是词与词之间的共现关系，即$P(w_{t+j}|w_t)$，其中$$j\in \{\pm{1}, \cdots, \pm{k}\}$$。以$k=2$为例，跳元模型也分为输入层、词向量层和输出层：
1. 输入层是当前时刻$w_t$的独热编码，通过矩阵$E$投射到隐含层；
2. 隐含层向量为$$w_t$$的词向量$$\pmb{v}_{w_t}=E^T_{w_t}$$;
3. 根据$$\pmb{v}_{w_t}$$，输出层利用线性变换矩阵$$E'$$对上下文窗口内的词进行预测：
$$P(c|w_t) = \frac{exp(\pmb{v}_{w_t} \cdot \pmb{v}^{'}_c)}{\sum_{w' \in \mathbb{V}}{exp(\pmb{v}_{w_t}\cdot \pmb{v}^{'}_{w'})} }$$
其中$$c\in \{ w_{t-k}, \cdots, w_{t-1}, w_{t+1},  \cdots, w_{t+k} \}$$。



#### 负采样优化

通过优化分类损失对CBOW和Skip-gram模型进行训练，需要估计的参数是$\theta = \{E, {E'}\}$。给定长度为$T$的词序列$w_1 w_2 \cdots w_T$，CBOW的负对数似然损失函数为：

$$\mathcal{L}(\theta) = -\sum_{t=1}^T logP(w_t|\mathcal{C}_t)$$


式中$$\mathcal{C}_t=\{w_{t-k}, \cdots, w_{t-1}, w_{t+1}, \cdots, w_{t+k}\}$$

Skip-gram模型的负对数似然损失为：

$$\mathcal{L}(\theta) = -\sum_{t=1}^T\sum_{-k\leq j\leq k,j \neq 0} logP(w_{t+j}|w_t) $$


当词表规模较大时，这类模型的训练会受到输出层概率归一化计算效率的影响，负采样[3]技术提供了一种全新的视角，给定词与上下文，最大化两者的共现概率，这样一来，问题就被简化为一个二分类问题，从而避免了大词表上的归一化计算。令$P(D=1|w,c)$表示$c$与$w$共现的概率：

$$P(D=1|w,c) = \sigma(\pmb{v}_w \cdot \pmb{v}_{c}^{'})$$

那么，两者不共现的概率为：

$$P(D=0|w,c) = 1- P(D|w,c) = \sigma(-\pmb{v}_w \cdot \pmb{v}_{c}^{'})$$

在CBOW模型中$$\{w_t,c\}$$是正样本，通过对$$w_t$$进行负采样和$$c$$构建负样本，构建二分类loss进行训练；在Skip-gram中，$$\{w_t,w_{t+j}\}$$为正样本，负样本取K个不出现在$w_t$上下文窗口内的词。



### GloVe
基于神经网络的词向量预训练方法本质上是利用文本中心词与局部上下文的共现信息作为自监督学习信号，另一类用于估计词向量的方法是基于矩阵分解的方法，这里方法主要是在整个预料上进行统计分析，获取全局统计信息“词上下文”共现矩阵，然后利用奇异值分解对矩阵进行降维。GloVe结合矩阵分解的思想来克服word2vec只关注局部特性的缺点。


Glove的基本思想是利用词向量对“词-上下文”共现矩阵进行预测，从而实现隐式的矩阵分解。首先构建共现矩阵$M$，其中$M_{w,c}$表示词$w$与上下文$c$在受限窗口大小内的共现次数。GloVe模型在构建M的过程中进一步考虑了$w$和$c$的距离，认为距离远的$(w,c)$对于全局共现次数的贡献比较小，因此采用共现距离进行加权的计算方式：

$$M_{w,c} = \sum_{i}\frac{1}{d_{i}(w,c)}$$

式中，$d_{i}(w,c)$表示第$i$次共现发生时，$w$与$c$之间的距离。构建完共现矩阵$W$之后，就可以使用词向量对$M$中的元素进行回归拟合，具体形式：

$$\pmb{v}_w^T \pmb{v}_c^{'} + b_w + b_c^{'} = logM_{w,c}$$

其中$$\pmb{v}_w^T$$和$$\pmb{v}_c^{'}$$是$w$和$c$的向量表示，$b_w$和$b_c$是对应的偏置，对该回归问题进行求解即可获得词和上下文的向量表示。Glove模型的回归损失函数为：

$$\mathcal{L}({E,E',b,b'};M) = \sum_{(w,c)\in \mathbb{D}} f(M_{w,c})(\pmb{v}_w^T \pmb{v}_c^{'} + b_w + b_c^{'} - logM_{w,c})^2$$


其中，$f(M_{w,c})$表示每一个$(w,c)$样本的权重。样本权重与其共现次数相关，共现次数很少的样本通常被认为含有较大的噪声，所蕴含的有用信息相对频繁共现的样本也更少，因此给与较低的权重，同时也会避免给与过高的权重，因此Glove采用以下分段函数进行加权：

$$f(M_{w,c}) = \begin{cases}
(M_{w,c}/\theta)^{\alpha}, & M_{w,c} \leq \theta \\
1, & others
\end{cases}$$


当$$M_{w,c}$$不超过阈值$$\theta$$时，$$f(W_{w,c})$$的值随$$M_{w,c}$$递增且小等1，当超出阈值时恒等1，$$\alpha$$用来控制增长速率。


### 参考文献
[1] Mikolov T, Chen K, Corrado G, et al. Efficient estimation of word representations in vector space[J]. arXiv preprint arXiv:1301.3781, 2013.
[2] Pennington J, Socher R, Manning C D. Glove: Global vectors for word representation[C]//Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014: 1532-1543.
[3] Mikolov T, Sutskever I, Chen K, et al. Distributed representations of words and phrases and their compositionality[J]. Advances in neural information processing systems, 2013, 26.