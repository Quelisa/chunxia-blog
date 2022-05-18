---
title: 跨语言预训练模型
date: 2022-03-18 19:30:17
tags:
- PLM
- cross-lingual
categories: 
- 跨语言模型
mathjax: true
---

目前在做国际化业务场景中的意图识别，主要探索的语种包括中文、英文、日文、阿文、西文和粤语的场景中。这一篇主要分享对于跨语言模型的调研，了解跨语言模型的现状，之后的文章会分享我自己在现有多语言模型上进行的训练过程，以及优化效果。
<!--more-->


### 跨语言模型调研

众所周知，在bert推出后不久，就推出了多语言bert，简称mbert。mbert与bert的唯一区别在于训练语料的不同，mbert支持104种语言，采用了110k的共享词表。虽然mbert在多语言效果上差强人意，但是很显然在提高多语言表达能力上，预训练模型还有很多工作可以做！借鉴来自机器翻译的经验和自然语言语义对齐词嵌入的经验，在mbert之后陆续提出了XLM、XLM-R、LaBSE、InfoXLM、XLM-E、VECO以及ERNIE-M等预训练多语言模型。除了上述采用encoder的跨语言模型，还有关注与内容生成的序列到序列的跨语言模型MASS和mBART。下面主要讲解下这几个模型的核心思想和架构。


#### XLM[1]
focebook在2019年提出了XLM模型，XLM主要是在bert的基础上，预训练借鉴了机器翻译中的翻译语言模型使用MLM+TLM或者CLM+MLM交替进行预训练。其中TLM是使用平行语料的监督方法，MLM和CLM是使用单语训练的无监督方法。同时为了平衡多种语言的分布差异，XLM在训练时采用一个缩放的分布对不同语种进行采样，以提高低资源语言的采样数。


#### XLM-R[2]
随后，focebook在2020年又提出支持100种语言的XLM-R模型，XLM-R采用了XLM的训练方法并使用RoBerta模型。XLM-R使用CommonCrawl数据集进行训练，其数据量较mbert使用的wiki有很大的增加，尤其是在低资源的语种上。实验表明XLM-R相较mbert在XNLI数据集上有很大的提升，在低资源的语种上提升也非常显著。在论文中也首次对影响多语言模型效果的因素进行了分析。主要是从高低资源的权衡以及语言采样和词汇大小的影响。论文通过实验证明了增加相似的语种进行训练可以提高该语种的效果，但是多语种会导致模型的参数增加，增加训练和推理成本。


#### LaBSE[3]
Google在2020年也推出了自己的跨语言模型LaBSE。LaBSE是基于bi-encoder结构的bert模型，LaBSE支持109种语言，使用MLM+TLM基于平行语料翻译排序任务进行预训练。LaBSE在Tatoeba数据集上获得83.7%的准确率，在此之前的最佳模型得分是65.5%。有意思的是Tatoeba数据集中的30+语种，并没有参与到LaBSE的训练，但是模型依然有较好的表现，展现了LaBSE模型zero-shot的能力。最后文章分析指出，使用预训练的bert可以减少对平行语料的需求。 使用CommonCrawl的数据训练的结果比使用wiki训练的效果好，因为CommonCrawl的数据量更大，噪声也更小。还有一个有趣的发现，使用预训练对比数据选择模型（CDS）选择出来的训练数据要比直接在原始数据上进行训练效果更好，这表明LaBSE对于数据的选择是敏感的。有关这一点我觉得大多数的模型对于数据的选择可以说都是敏感的。这类问题应该归属于领域泛化问题。



#### XLM-E[4]
XLM-E是微软提出的跨语言模型，它主要基于ELECTRA模型训练的，所以它是一种对抗生成式的模型。它的预训练包括两个任务，一个是多语言替换词检测（MRTD），另一个是翻译替换词检测（TRTD）。替换词检测是在单语语料上，针对每一个句子进行mask后让生成器生成对应的mask位置上的词，然后将生成的词替换原来的句子中的词，最后让判别器判别每个词是生成的还是原来的。翻译替换词检测是在翻译的平行语料句子对上，mask掉每个句子中的一部分词，让生成器去生成mask对应位置的上的词，用生成的词去替换原来句子中对应的词，最后再让判别器去判别该词是否是原来的词。


#### InfoXLM[5]
InfoXLM是微软开源的跨语言模型，它的创新在于受到信息论和对比学习的启发，设计了一个新的跨语言预训练任务cross-lingual contrast（XLCO）。InfoXLM的预训练的任务除了XLCO还有MMLM和TLM。其MMLM和TLM的损失函数是从词级别互信息的角度建立的。MMLM针对是单语语料输入的文本进行单词的mask来预测被mask的单词，最大化文本和被mask掉的单词的互信息，根据InfoNCE中的理论构建对比loss，$$L_{MMLM}$$。同理，TLM只是针对双语翻译的语料，对于同一对句子$$(c_1,c_2)$$，$$x_1$$是$$c_1$$中被mask掉的单词，目标是最大化句子对和被mask掉的词的互信息，这个互信息可以被拆解成两个部分，一部分是$$c1$$和$$x_1$$的互信息，另一部分是$$c_2$$和$$x_1|c_1$$的互信息，前一部分相当于是MMLM，后一部分则比较有趣，它其实是使用不同的语言来预测被mask掉的词，因此提高了模型跨语言迁移的能力，其loss为$$L_{TLM}$$。而XLCO的损失函数是最大化平行句子嵌入，是从句子级别的互信息角度建立的，其loss为$$L_{XLCO}$$。最后InfoXLM的训练loss为$$L = L_{MMLM} + L_{TLM} + L_{XLCO}$$


#### ERNIE-M[6]
ERNIE-M是百度基于自己的预训练模型ERNIE在多语言语料上进行训练的跨语言模型。ERNIE-M主要是在预训练的过程中加入了机器翻译中经典的回译方法，对单语数据生成伪平行语料，来解决平行语料不足的问题。ERNIE-M的预训练任务主要包括交叉注意力掩码语言模型（CAMLM）、回译掩码语言模型（BTMLM）、多语言掩码语言模型（MMLM）、翻译语言模型（TLM）。


这里主要介绍一下CAMLM和BTMLM。下面放两张有关预训练语言模型CAMLM和TLM与MMLM区别的图片，直观感受一下CAMLM的语言对齐的方式。CAMLM中对于被mask的词的词义推理仅依赖平行语料的的另一个句子，不能依赖自己所在的句子本身去推断，而MMLM和TLM则可以。
<img src="https://github.com/Quelisa/picture/raw/main/LM/CAMLM.png" width="80%" height="50%">


对于BLMLM，其训练过程主要分为两个阶段：第一个阶段用源句子加掩码做为目标语言的句子，生成目标语言的翻译数据；第二个阶段将生成的伪平行语料输入，并对源句子进行掩码，训练掩码的输出。具体过程如下图：
<img src="https://github.com/Quelisa/picture/raw/main/LM/BTMLM.png" width="50%" height="60%">

整体来说ERNIE-M效果还不错，在XNLI数据集上ERINE-M的效果比InfoXML要好。


### 参考文献

[1] Lample G, Conneau A. Cross-lingual language model pretraining[J]. arXiv preprint arXiv:1901.07291, 2019.
[2] Conneau A, Khandelwal K, Goyal N, et al. Unsupervised cross-lingual representation learning at scale[J]. arXiv preprint arXiv:1911.02116, 2019.
[3] Feng F, Yang Y, Cer D, et al. Language-agnostic bert sentence embedding[J]. arXiv preprint arXiv:2007.01852, 2020.
[4] Chi Z, Huang S, Dong L, et al. XLM-E: cross-lingual language model pre-training via ELECTRA[J]. arXiv preprint arXiv:2106.16138, 2021.
[5] Chi Z, Dong L, Wei F, et al. InfoXLM: An information-theoretic framework for cross-lingual language model pre-training[J]. arXiv preprint arXiv:2007.07834, 2020.
[6] Ouyang X, Wang S, Pang C, et al. ERNIE-M: enhanced multilingual representation by aligning cross-lingual semantics with monolingual corpora[J]. arXiv preprint arXiv:2012.15674, 2020.