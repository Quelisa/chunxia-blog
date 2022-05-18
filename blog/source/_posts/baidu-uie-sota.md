---
title: 信息抽取新范式：百度UIE刷新13个数据集SOTA
date: 2022-05-24 19:33:25
tags:
- Information Extraction
- SOTA
categories: 
- 信息抽取
---

百度最近提出的统一文本到结构生成的框架UIE（Universal Information Extraction），将实体、关系、事件和情感四大抽取任务统一建模，并在13个数据集上达到了SOTA。令人振奋的不仅是效果上的提升，还有对于信息抽取革命性的统一！下面看看UIE是怎么做的吧！

<!--more-->

### Task-specialized IE VS. Universial IE

目前大多数的信息抽取都是分任务进行的，针对一个任务建立一个模型进行任务学习，这给信息抽取带来了很大的复杂度，百度提出一个统一文本到结构生成的框架UIE，将实体信息抽取、关系抽取、事件抽取和敏感性抽取统一建模，实现一个模型抽取多种信息。下图是这两种信息抽取方式比较直观的对比。

<img src="https://github.com/Quelisa/picture/raw/main/information-extraction/task-IEvsUIE.png" width="50%" height="60%">

综上可以将所有的信息抽取进行统一建模为两个步骤：（1）抽取点，可以是实体、事件等；（2）建立联系，将收取到的点信息之间建立关联。


### Universial IE 架构
UIE是如何统一建模的呢？首先，因为之前的信息抽取都是分任务的，每一个任务都有自己的输出表示，要建立一个统一的模型在结构输出上首先要有一个结构规范，为此论文中提出了一种结构化抽取语言（SEL），对信息结果进行统一管理。其次，虽然是统一的信息抽取模型，但是我们希望它能够自适应的输出所需要的任务的结果，而不是给出所有的结果，因此论文中提出了在此基础上提出了结构化模式提示器（SSI）来控制对不同任务的生成需求。综上就是UIE结构的核心，它可以表达为：SSI+Text->SEL，
UIE架构图如下：
<img src="https://github.com/Quelisa/picture/raw/main/information-extraction/UIE.png" width="80%" height="70%">


#### SEL：结构化抽取语言

举例如图所示：
<img src="https://github.com/Quelisa/picture/raw/main/information-extraction/SEL.png" width="40%" height="60%">


#### SSI：结构化模式提示器

SSI的本质一个基于schema的prompt机制，用于控制不同的生成需求：在Text前拼接上相应的Schema Prompt，输出相应的SEL结构语言。例如：
1. 实体抽取：[spot] 实体类别 [text]
2. 关系抽取：[spot] 实体类别 [asso] 关系类别 [text]
3. 事件抽取：[spot] 事件类别 [asso] 论元类别 [text]
4. 观点抽取：[spot] 评价维度 [asso] 观点类别 [text]



### UIE 训练

UIE采用预训练+微调的方式进行训练，预训练主要针对：
1. Text-to-Structure Pre-training：构建基础的文本到结构的映射能力，使用text-to-struct的平行语料（SSI，TEXT，SEL）进行训练；
2. Structure Generation Pre-training：训练具备SEL语言的结构化能力，使用包含SEL语法的record数据（None，None，SEL）进行训练；
3. Retrofitting Semantic Representation：训练具备基础的语义编码能力，使用原始文本（None，TEXT，TEXT）进行训练。


微调阶段为解决自回归teacher-forcing的暴露偏差，构建了拒识噪声注入的模型微调机制。


### 展望
少样本实验可以发现，大规模异构监督预训练可以学习通用的信息抽取能力，使模型具有更好小样本学习能力。UIE的提出只是信息抽取统一建模的开始，论文的作者也表明未来希望将类似指代消解等任务也在信息抽取中进行统一建模学习。我感觉UIE本质上还是走的大规模预训练训练+Prompt+多任务微调的范式，但是不得不承认这种范式确实已经成为一种主流。



### 参考文献
[1] Lu Y, Liu Q, Dai D, et al. Unified Structure Generation for Universal Information Extraction[J]. arXiv preprint arXiv:2203.12277, 2022.