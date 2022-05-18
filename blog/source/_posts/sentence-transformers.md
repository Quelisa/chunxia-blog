---
title: 有关句嵌入模型的整理
date: 2022-01-15 16:41:14
tags:
- 文本匹配
- Representation Learning
categories: 
- Sentence Embedding
mathjax: true
---

句向量嵌入也就是sentence embedding对于理解文本语义有很重要的作用，很多任务都需要用到sentence embedding，比如文本相似度的计算、文本检索、聚类、文本语义挖掘等。因此训练具有良好特性句向量嵌入的模型十分重要。本篇主要从sentence-transformer开源的句向量嵌入模型[1]选取几个有代表性的模型，进行分析，了解句向量发展的主流技术和面临的挑战。

<!--more-->


### 无监督句嵌入模型
收到词向量的启发，自然地想到在自然语言处理中由于词的粒度较小，表达的信息比较离散，很多的场景需要用到句向量进行建模。而无论是将词映射的向量空间还是将句子映射到向量空间，在建模时我们都希望语义相似的句子在空间彼此靠近，而语义差别大的句子在空间中的距离彼此疏远，同时尽可能的保持空间向量分布的均匀性。一般对于BERT类型的语言模型，句向量的表示可以使用其[CLS]标志位上的向量进行表示，也可以对最后一层或者多层中所有输出采样（平均值采样、最大值采样等）后的结果进行表示。为什么采用这些数据向量而它们为什么具有语义表达的能力？它们在表示语义的时候又存在哪些问题？这就是主流的句嵌入模型研究的出发点。


#### Sentence-BERT[2]
对于计算语义文本相似度，BERT支持同时输入句子对进行计算，但是这样的结构在推理的时候，需要将所有句子的组合都经过网络计算，计算量非常大。Sentence-BERT也称SBERT采用双塔结构，分别对每个句子进行embedding，然后将embedding存起来，再进行单独的相似度计算，可以极大的提高推理速度。缺点是相对BERT的句子对计算会带来语义匹配度上的损失。

SBERT的模型结构如图，图中左侧是一个训练的结构，图中右侧是一个模型的推理的结构，训练目标针对分类任务使用了交叉熵损失函数、针对相似度计算使用了均方差损失函数以及还是用了对比学习中的Triplet loss。
<img src="https://github.com/Quelisa/picture/raw/main/sentence-embedding/SBERT.png" width="60%" height="50%">

#### BERT-flow[3] 和 BERT-whitening[4]
BERT-flow是字节提出的基于BERT模型的改进。考虑到未经过fine-tune的BERT在语义表达上的效果还不如GloVe，为此探索BERT的语义表达能力瓶颈，是因为BERT没有学习到语义表达的能力还是BERT的语义表达能力没有被充分地发掘？根据以往的研究可以知道BERT的词嵌入表现出各向异性，空间分布呈现锥形，高频的词分布较为集中靠近锥尖，低频词距离较远原理锥尖。因此假设BERT在文本嵌入方面可能也存在类似的问题，于是作者提出了一种映射方式，将BERT的训练出的Embeddings通过一个可逆映射到均匀分布的空间，这个映射采用的flow模型，因此叫BERT-flow。


BERT-whitening是追一科技提出的模型改进，他认为克服BERT本身的句嵌入空间向量集合各向异性不需要一个flow模型，只要一个线性变换就可以搞定。文本相似度计算是采用cosine计算的，cosine的计算公式是向量內积再做归一化，这个等式成立的前提是标准坐标基空间。可以通过统计学上假设，如果是标准正交基，那么它对应的向量应该表现出各向同性，而BERT句向量存在各向异性，所以认为它所属的坐标系不是标准正交基。作者指出flow模型的表达能力很弱而且模型参数很大，用一个flow模型有些浪费，而根据前面的假设推理，作者采用数据挖掘中的白化操作对向量进行线性变化，变化为一个标准的均值为0，协方差矩阵为单位阵的正态分布。


#### Constrastive Tension[5]
Constrastive Tension也称CT，这篇文章主要研究了对于STS任务transformer架构每一层的句嵌入向量的特点，然后针对句嵌入向量层敏感的特性采用对比方式进行改进。下面是几个主流模型每一层对于语义的敏感程度。
<img src="https://github.com/Quelisa/picture/raw/main/sentence-embedding/transformer-layer-sts.png" width="60%" height="80%">
可以看到在最后一层的时候语义表征反而降低了。为了对抗这种语义表达的损失，CT的核心是采用两个单独的模型，初始化参数相同，对最后一层的进行平均采用，采用对比损失，最大化相同句子的相似度并最小化不同句子的相似度。感觉本质上就是一种对比训练。


#### SimCSE[6]
SimCSE真的YYDS！完全没有花里胡哨的东西，还非常能打，最近一直在用SimCSE的方法进行对比训练，效果都有提升。SimCSE主要采用非对比学的方法，采用InfoNCE loss。创新点在于正例的选择，没有用各种奇奇怪怪的数据增强方式，仅用dropout，因为在dropout不为0时，同样的句子分别输入网络得到的句向量就是不同的。负例采用同一个batch中的其他句子。无监督学习，数据获取容易，效果好！


SimCSE还做了一个监督的对比训练，其实主要是用类似于推理的数据，三个句子同时输入，前两个保持一致，最后一个句子和前面的矛盾，用来做强约束。效果比无监督的好，但是实际使用中感觉无监督更好用一些，毕竟构造高质量有监督的数据还是比较麻烦的。


#### TSDAE[7]
TSDAE是基于Transformer结构的序列噪声自编码模型，模型结构如下：
<img src="https://github.com/Quelisa/picture/raw/main/sentence-embedding/TSDAE.png" width="30%" height="40%">
它主要是在模型的输入中将个别词进行删除或者替换，通过encoder编码成一个固定长度的向量，然后通过decoder重建原来的输入。TSDAE在多个数据集上的测试效果都优于CT、SimCSE和BERT-flow等。



#### LaBSE[8]
LaBSE是跨语言的预训练模型，它支持109中语言，具有跨语言迁移的效果。它采用基于BERT的双塔模型架构。
<img src="https://github.com/Quelisa/picture/raw/main/sentence-embedding/LaBSE.png" width="50%" height="40%">

它使用对比学习进行。其loss是在InfoNCE loss的基础上进行了修改，加入了margin，可以更好地拉开正例和负例的距离。同时对比学习中负例的样本数越大，对比效果越好，所以LaBSE在训练是采用了交叉加速负采样（就是使用n个TPU，每个核中batch为8，从同一个batch采样负例的同时也从其他的核中采样负例），从而达到增加负例的效果。我感觉LaBSE另一个重大的贡献在于采用了大量而且丰富的多语种语料，才能达到如此优秀的跨语种迁移能力。


### 方法总结
目前主流的句向量模型都是在BERT或者BERT类型的预训练模型上，针对BERT预训练模型语义表达的痛点进行改进，例如通过空间映射到均匀空间解决BERT句向量空间各向异性的问题；还有采用对比学习的方法，通过拉近相似文本的空间向量表示，疏远不同样本的空间向量表示等。究其本质，都是默认把文本编码成句向量，希望空间的句向量表达在空间分布均匀，在语义理解上根据向量的距离进行识别。

### 参考文献
[1] https://www.sbert.net/
[2] Reimers N, Gurevych I. Sentence-bert: Sentence embeddings using siamese bert-networks[J]. arXiv preprint arXiv:1908.10084, 2019.
[3] Li B, Zhou H, He J, et al. On the sentence embeddings from pre-trained language models[J]. arXiv preprint arXiv:2011.05864, 2020.
[4] Su J, Cao J, Liu W, et al. Whitening sentence representations for better semantics and faster retrieval[J]. arXiv preprint arXiv:2103.15316, 2021.
[5] Carlsson F, Gyllensten A C, Gogoulou E, et al. Semantic re-tuning with contrastive tension[C]//International Conference on Learning Representations. 2020.
[6] Gao T, Yao X, Chen D. Simcse: Simple contrastive learning of sentence embeddings[J]. arXiv preprint arXiv:2104.08821, 2021.
[7] Wang K, Reimers N, Gurevych I. Tsdae: Using transformer-based sequential denoising auto-encoder for unsupervised sentence embedding learning[J]. arXiv preprint arXiv:2104.06979, 2021.
[8] Feng F, Yang Y, Cer D, et al. Language-agnostic bert sentence embedding[J]. arXiv preprint arXiv:2007.01852, 2020.