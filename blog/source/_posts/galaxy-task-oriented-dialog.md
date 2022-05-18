---
title: GALAXY：面向任务型对话的生成式预训练模型
date: 2022-01-11 16:43:21
tags:
- PCM
- Dialog
categories: 
- 任务型对话
---

GALAXY[1]是阿里提出的一个针对任务型对话的预训练对话模型（Pre-trained Conversation Model，PCM），它引入了半监督学习对对话策略进行学习，在In-Car，MultiWOZ2.0和MultiWOZ2.1数据集上达到了SOTA。

<!--more-->

### 任务型对话技术现状
任务型对话一般分为三个阶段：（1）自然语言理解；（2）对话策略以及（3）对话生成。对话策略主要是用来连接自然语言理解到自然语言生成部分的逻辑。任务型对话有两种主流实现，一种是pipeline的实现方式，另一种是端到端的实现。随着预训练语言模型的兴起，研究人员更多的开始研究基于预训练的对话模型。以往的对话模型更多的关注自然语言理解和自然语言生成部分，从而忽视了任务型中很重要的对话策略，GALAXY强调了对话策略的重要性，基于UniLM模型框架建立了新的预训练对话模型。


### 基于预训练对话模型（PCM）的任务型挑战

1. 缺乏大量连续对话数据
2. 大部分对话数据没有标签
3. 自监督的标签使得无法对对话动作空间进行探索，提取知识有限


### GALAXY的做了啥

1. 从现有的对话数据集中清洗出有标签数据集UniDA和无标签数据集UnDial
2. 基于UniLM建立了优化Loss为对话选择、对话生成、对话预测和KL散度之和的半监督预训练对话模型

下图是GALAXY的架构:
<img src="https://github.com/Quelisa/picture/raw/main/Dialogue/GALAXY.png" width="80%" height="50%">

可以看到左侧为模型的输入包括对话文本，对话角色对话伦次以及词在文本中的位置信息，右侧是模型的主结构主结构中将分为三种网络连接，一种是context到context的自关注全连接网络，一种是conext到response关注的全连接网络，另一种是双向的context到response全连接网络，针对context的自关注计算对话预测损失和KL散度（这里之所以加一个KL散度是为了是模型的泛化能力更好），针对context到response的单向网络连接计算对话生成损失，针对全连接关注计算对话选择的损失。预训练模型的损失是上述损失之和，微调的损失是对话选择、对话生成和对话预测损失之和。其中只有对话预测要求有监督的数据进行训练，所以说GALAXY是一种半监督的预训练对话模型。


### 参考文献
[1] He W, Dai Y, Zheng Y, et al. GALAXY: A Generative Pre-trained Model for Task-Oriented Dialog with Semi-Supervised Learning and Explicit Policy Injection[J]. arXiv preprint arXiv:2111.14592, 2021.