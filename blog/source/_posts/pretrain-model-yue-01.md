---
title: 粤语模型训练系列之数据收集
date: 2022-03-10 11:13:48
tags:
- 粤语
- 数据收集与清洗
categories: 
- 跨语言模型
---

最近在做粤语模型的训练，训练的第一步，当然就是要收集数据啦！这一篇主要分享粤语语料的收集和数据处理。
<!--more-->


### 粤语语料收集
提到多语言语料收集，Wikipedia的数据绝对是第一个想到，Wikipedia有一个专门备份的网站[2]，提高各个语种信息的备份，所有的网页内容都可以在此下载。我下载的是20220501的数据，粤语wiki的大小是88M。我在huggingface的数据集中搜到了中文和粤语的翻译平行语料：x-tech/cantonese-mandarin-translation[3]。网上有关粤语的语料有一个叫《香港二十世纪中期粤语语料库》[4]，该语料库需要注册登录，是一个检索式的语料库且不可下载。除此之外，可以直接拿到的数据少之又少0.0。所以就需要用到爬虫工具爬取一些粤语的微博、博客、电影对话等。记录一下粤语的网站：
1. http://www.hkcna.hk/index.jsp?channel=2803
2. https://weibo.com/u/6227634537?refer_flag=1005055013_
3. https://subanana.com/ 是一个AI自动生成粤语字幕的网站
4. https://commonvoice.mozilla.org/zh-CN/datasets


### 网络爬虫介绍
网上搜索了一堆的爬虫应用和脚本，真正免费而又好用的没几个。正好淘一淘，也学习一下爬虫。找到了几个看起来还不错的爬虫服务，但是用起来体验一般，不过还是可以记录一下，毕竟做的还不错，也是一种商业化渠道。
1. https://app.diffbot.com/
2. https://commoncrawl.org/the-data/get-started/
3. https://webz.io/
4. https://www.parsehub.com/
5. https://www.scrapingbee.com/


### 语料数据清洗
前面提到的wiki数据还是带有网络格式的数据，并不能直接用于模型训练，而且数据质量并没有经过筛选。下面介绍一个好用的工具GENSIM[5]，用来直接抽取wiki中的文本数据。下面的代码可以抽取出wiki的内容，但是它会去掉所有的标点符号用空格代替。

```python
from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    output = open('wiki.txt', 'w', encoding='utf8')
    wiki = WikiCorpus("wiki-xxx.xml.bz2", dictionary={})
    for text in wiki.get_texts():
        output.write(" ".join(text) + "\n")
    output.close()		 
```

后续参考苏神的代码，重新对wiki数据进行了整理，清洗代码在代码仓库：https://github.com/Quelisa/data_cleaner.git

### 总结

因为粤语的收集难度确实大于中文的收集难度，并且我们已经知道这两个模型都支持中文并且都已经在大规模的中文数据集上进行了训练，所以我打算利用一些现成的中文数据集和收集到的粤语语料分别进行单语种的训练，然后主要通过收集到的平行语料进行中文和粤语的对齐训练，因为可用的平行语料不是很多所以可以通过翻译的方式获得。语料收集是算法训练中看似枯燥，却又极其重要的一步，语料的质量直接关系到模型效果的好坏，可谓是失之毫厘谬以千里。收集并设计好的语料库，高效的开发工具和科学的处理方法同样重要！第一次真正意义上感受到了big data和数据质量的重要性！


目前专门针对粤语的预训练语言模型只有哈工大讯飞联合实验室发布的CINO (Chinese minority PLM)[1]，理论上具有和粤语语系相近的多语言模型也具有一定的粤语识别能力。在LaBSE的论文中看到,虽然没有用到粤语训练，但是粤语的识别效果也比较可观。所以打算在CINO和LaBSE上针对粤语再进行训练，以提高粤语的意图识别能力。


### 参考文献
[1] https://github.com/ymcui/Chinese-Minority-PLM
[2] https://dumps.wikimedia.org/zh_yuewiki/
[3] https://huggingface.co/datasets/x-tech/cantonese-mandarin-translations
[4] https://hkcc.eduhk.hk/v1/introduction.html
[5] https://radimrehurek.com/gensim/apiref.html#api-reference