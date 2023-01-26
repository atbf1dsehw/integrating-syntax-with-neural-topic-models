# This directory contains the data package.

Huggingface is extensively used in terms of getting the datasets.

## List of datasets and their origin, with the citation:

### yelp_review_full [(click_here)](https://huggingface.co/datasets/yelp_review_full)

Dataset Summary: The Yelp reviews dataset consists of reviews from Yelp. It is extracted from the Yelp Dataset Challenge 2015 data. The Yelp reviews full star dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the Yelp Dataset Challenge 2015. It is first used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).

-    Citation: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).
-   Labels: 1, 2, 3, 4, 5
-   Data Fields: ['text', 'label']

### imdb [(click_here)](https://huggingface.co/datasets/imdb)

Dataset Summary: Large Movie Review Dataset. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.

-    Citation:
```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```
-   Labels: 0, 1
-   Data Fields: ['text', 'label']
-   Data Splits: {'train': ['train'], 'test': ['test'], 'unsupervised': ['unsupervised']}


### ag_news [(click_here)](https://huggingface.co/datasets/ag_news)

Dataset Summary: AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining (clustering, classification, etc), information retrieval (ranking, search, etc), xml, data compression, data streaming, and any other non-commercial activity. For more information, please refer to the link http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .

The AG's news topic classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the dataset above. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).

-    Citation:
```bibtex
@inproceedings{Zhang2015CharacterlevelCN,
  title={Character-level Convolutional Networks for Text Classification},
  author={Xiang Zhang and Junbo Jake Zhao and Yann LeCun},
  booktitle={NIPS},
  year={2015}
}
```
-   Labels: 0, 1, 2, 3
-   Data Fields: ['text', 'label']
-   Data Splits: {'default': ['train', 'test']}

### 20newsgroup [(click_here)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups)

Dataset Summary: This is a dataset of 20 newsgroups posts on 20 topics. The original dataset was collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews paper, though he does not explicitly mention this collection. The 20 topics are:

-   alt.atheism
-   comp.graphics
-   comp.os.ms-windows.misc
-   comp.sys.ibm.pc.hardware
-   comp.sys.mac.hardware
-   comp.windows.x
-   misc.forsale
-   rec.autos
-   rec.motorcycles
-   rec.sport.baseball
-   rec.sport.hockey
-   sci.crypt
-   sci.electronics
-   sci.med
-   sci.space
-   soc.religion.christian
-   talk.politics.guns
-   talk.politics.mideast
-   talk.politics.misc
-   talk.religion.misc

-    Citation:
```bibtex
@incollection{LANG1995331,
title = {NewsWeeder: Learning to Filter Netnews},
editor = {Armand Prieditis and Stuart Russell},
booktitle = {Machine Learning Proceedings 1995},
publisher = {Morgan Kaufmann},
address = {San Francisco (CA)},
pages = {331-339},
year = {1995},
isbn = {978-1-55860-377-6},
doi = {https://doi.org/10.1016/B978-1-55860-377-6.50048-7},
url = {https://www.sciencedirect.com/science/article/pii/B9781558603776500487},
author = {Ken Lang},
}
```
-   Labels: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
-   Data Fields(subset): ['train', 'test', 'all']
-   Remove(remove): ['headers', 'footers', 'quotes']

```python
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
```

### rotten_tomatoes [(click_here)](https://huggingface.co/datasets/rotten_tomatoes)

Dataset Summary: Movie Review Dataset. This is a dataset of containing 5,331 positive and 5,331 negative processed sentences from Rotten Tomatoes movie reviews. This data was first used in Bo Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.'', Proceedings of the ACL, 2005.

-    Citation:
```bibtex
@InProceedings{Pang+Lee:05a,
  author =       {Bo Pang and Lillian Lee},
  title =        {Seeing stars: Exploiting class relationships for sentiment
                  categorization with respect to rating scales},
  booktitle =    {Proceedings of the ACL},
  year =         2005
}
```
-   Labels: 0, 1
-   Data Fields: ['text', 'label']
-   Data Splits: {'train': ['train'], 'test': ['test'], 'validation': ['validation']}


### amazon_polarity [(click_here)](https://huggingface.co/datasets/amazon_polarity)

Dataset Summary: The Amazon reviews dataset consists of reviews from amazon. The data span a period of 18 years, including ~35 million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review.

Citation1: McAuley, Julian, and Jure Leskovec. "Hidden factors and hidden topics: understanding rating dimensions with review text." In Proceedings of the 7th ACM conference on Recommender systems, pp. 165-172. 2013.

Citation2: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).

-    Labels: 0, 1
-    Data Fields: ['title', 'content', 'label']

### ccdv/govreport-summarization [(click_here)](https://huggingface.co/datasets/ccdv/govreport-summarization)

```bibtex
@misc{huang2021efficient,
      title={Efficient Attentions for Long Document Summarization}, 
      author={Luyang Huang and Shuyang Cao and Nikolaus Parulian and Heng Ji and Lu Wang},
      year={2021},
      eprint={2104.02112},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
```