---
title: 预训练模型RoFormer浅析
date: 2022-05-06 15:51:19
tags:
- PLM
categories: 
- 预训练语言模型
mathjax: true
---


近期我司发布了自研预训练语言模型RoFormerV2[1]，RoFormer的核心是基于苏神的“旋转位置编码（Rotary Position Embedding，RoPE）”[2]。今天就来学习一下RoFormer的进化过程！
<!--more-->

说实话，看完苏神在科学空间的博客中的公式推导，内心经历十分跌宕从兴致勃勃到一脸懵逼到竟然这样又到顶礼膜拜。但是既然要学习一下，就硬拿起来啃一下~苏神原文中的公式推导已经很好了，我在这篇文章主要分享一下我的理解过程吧！


### 追根溯源
大家都知道Transformer[1]的编码除了有词id，还需要有词的位置编码，因为attention无法区分不同位置的词。transformer的做法是将句子中所有词的绝对位置硬编码融入到输入中去。那么这种硬编码有什么优劣呢？首先编码简单是不用说的，比如一般是设置最大长度512，按照512*dim（向量维度）初始化位置向量即可。那有什么缺点呢？其实我们可以直观的感受的这种方式比较自然容易想到，但是也比较固定，不够灵活，我们的语言是千变万化的，绝对位置的编码限制了语言的自由组合，所以研究人员开始提出各种各样的相对位置编码。那么什么是相对的位置编码呢？其实也很直观，比如有一个输入的句子特别长，其实完全考虑整个句子的绝对位置编码得出的信息并没有那么多。相反那些相对位置比较近的词句反而可以提取出更多的语义特征，因此研究人员通过对带绝对位置的attention进行公式转换，截断长句位置信息，保留与该词位置较近的词的位置信息，当然方法是层出不群的。那么RoFormer做了什么呢？RoFoermer采用的RoPE旋转位置编码是一种融合了绝对位置编码和相对位置编码的编码方式。既保留了绝对位置信息，又增强了相对位置的信息。


### RoPE思想
RoPE假设给attention中的q，k增加绝对位置m，n信息后的函数$$f(q,m)$$和$$f(k,n)$$做点积的结果是一个之和相对位置(m-n)有关的函数$$g(q,k,m-n)$，即：$<f(q,m), f(k,n)> = g(q,k,m-n)$$然后作者使用了复数形式的向量计算进行推导出含有绝对位置形式的函数形式：$$f(q,m) = ||q||e^{i(\Theta (q) + m\theta)} = qe^{im\theta}$$。根据复数乘法的几何意义，该变换可以看做是向量的旋转，这也是RoPE旋转位置编码的由来。该公式可以写成矩阵相乘的形式进行计算。RoPE采用了Sinusoidal位置编码[3]的方案，因此RoPE对于位置的编码具有远程衰减的特性，如图可见。
<img src="https://github.com/Quelisa/picture/raw/eb694e51379595f70ff9544f07f3a8b28cd30b99/RoPE.png" width="50%" height="50%">


因为RoPE是使用绝对位置编码来实现相对位置编码，所以它不需要对Attention矩阵进行操作，可以直接应用到线性attention中。RoFormer[4]论文中的实验也表明了使用RoPE编码的RoFormer对于长文本有很好的捕捉能力，它可以很好的处理任意长的文本。


### RoFormerV2做了什么升级
苏神在自己的博客的标题中[5]用到了《RoFormerV2：自然语言理解的极限探索》，可以看做是对RoformerV2的一个定位。这里的极限是指什么呢？与大多数预训练模型追求超大超深的网络不同，RoFormerV2探索的是预训练模型在相同参数数量下性能的极限。RoFormerV2的主要改动就是简化模型建构，使其成为GLUE榜上前五名中参数数量最少的模型。RoFormerV2去掉了模型中的所有bias项，并且加大了训练数据从30多G加到280G（深度学习果然是数据为王呀！）。并且RoFormerV2是从零开始训练，为了抵抗去掉参数带来的效果损失，还增加了监督的多任务进行训练。


### 参考文献
[1] https://github.com/ZhuiyiTechnology/roformer-v2
[2] https://spaces.ac.cn/archives/8265
[3] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.
[4] Su J, Lu Y, Pan S, et al. Roformer: Enhanced transformer with rotary position embedding[J]. arXiv preprint arXiv:2104.09864, 2021.
[5] https://kexue.fm/archives/8998