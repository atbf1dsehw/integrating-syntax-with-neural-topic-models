import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


class TopicEval:
    """This is a class for evaluating topic models. 
    It returns a dictionary with the evaluation metrics.

    Attributes:
        vocab (dict): The vocabulary of the topic model.
        text (list): The list of documents used as reference for the evaluation.
        
    Methods:
        topic_coherence(self, metric: str = "c_v", topics: list): Returns the topic coherence of the topics (using CoherenceModel).
        topic_diversity(self, topics: list): Returns the topic diversity of the topics.
        semantic_purity(self, topics: list): Returns the syntax composition (stop words and punctuations) of the topics.
    """

    def __init__(self,
                 vocab: dict,
                 text: list) -> None:
        # vocab is a dictionary with the words as keys and the indices as values
        # converting the vocab to pandas dataframe with index and word columns
        vocab = {v: k for k, v in vocab.items()}
        self.vocab = pd.DataFrame.from_dict(vocab, orient='index', columns=['word'])
        self.vocab['index'] = self.vocab.index
        self.text = text
        self.dictionary = Dictionary(self.text)

    def get_topics(self, top_n: int, beta: np.array):
        topics = []
        for i in range(beta.shape[0]):
            indices = np.argsort(beta[i, :])[::-1]
            df = pd.DataFrame(indices[:top_n], columns=['index'])
            names = pd.merge(df, self.vocab[['index', 'word']], how='left', on='index')['word'].values
            topics.append(names.tolist())
        return topics

    def topic_coherence(self, metric: str, topics: list) -> float:
        cm = CoherenceModel(topics=topics,
                            texts=self.text,
                            dictionary=self.dictionary,
                            coherence=metric)
        return cm.get_coherence()

    def topic_diversity(self, topics: list) -> float:
        unique_words = set()
        for topic in topics:
            unique_words.update(topic)
        return len(unique_words) / (len(topics) * len(topics[0]))

    def semantic_purity(self, topics: list):
        syntax = stopwords.words('english')
        syntax_composition = 0
        for topic in topics:
            for word in topic:
                if word in syntax:
                    syntax_composition += 1
        return (1-syntax_composition / (len(topics) * len(topics[0]))) * 100
