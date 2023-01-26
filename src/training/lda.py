import os
import re
import logging
import pickle
import torch
from src.models.ldamallet import LdaMallet
from gensim.corpora.dictionary import Dictionary
from src.utils import TopicEval
from gensim.models import LdaModel

logger = logging.getLogger(__name__)


class LDATrainer:
    def __init__(self,
                 num_topics: int,
                 text: list,
                 train_dl: torch.utils.data.DataLoader,
                 vocab: dict,
                 save_path: str) -> None:
        train_text = []
        id2txt = {v: k for k, v in vocab.items()}
        self.vocab = vocab
        for batch in train_dl:
            bow = batch['bow']
            for doc in bow:
                doc_text = []
                # doc is a tensor of shape (vocab_size, )
                words = torch.nonzero(doc)
                words = words.squeeze(1)
                freqs = doc[words]
                for word, freq in zip(words, freqs):
                    word = id2txt[word.item()]
                    doc_text.extend([word] * freq.item())
                train_text.append(doc_text)
        self.text = text
        self.save_path = save_path
        self.num_topics = num_topics
        common_texts = train_text
        self.common_dictionary = Dictionary(common_texts)
        self.common_corpus = [self.common_dictionary.doc2bow(doc) for doc in common_texts]
        self.logger = {}

    def run(self):
        logger.info(f"Training started.")
        try:
            mallet_path = 'path/to/mallet'
            model = LdaMallet(mallet_path,
                              corpus=self.common_corpus,
                              num_topics=self.num_topics,
                              id2word=self.common_dictionary)
        except:
            # if mallet is not installed, use gensim's LDAModel. It is faster and usually as good or better.
            # (https://papers.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf)
            model = LdaModel(corpus=self.common_corpus,
                             id2word=self.common_dictionary,
                             num_topics=self.num_topics)
        topics = model.print_topics(-1)
        topic_words = [re.findall(r'"(.*?)"', topic[1]) for topic in topics]
        logger.info(f"Training completed. Now going for evaluation.")
        # beta is of shape (total_topics, vocab_size)
        eval = TopicEval(vocab=self.vocab, text=self.text)
        coherence = eval.topic_coherence(metric='c_v', topics=topic_words)
        topic_diversity = eval.topic_diversity(topics=topic_words)
        logger.info(f"Evaluation completed. Coherence scores are: {coherence}, topic diversity: {topic_diversity}")
        self.logger['topics'] = topic_words
        self.logger['tq'] = {'coherence': coherence,
                             'topic_diversity': topic_diversity}
        pickle.dump(self.logger, open(os.path.join(self.save_path, 'logger.pkl'), 'wb'))
        pickle.dump(self.text, open(os.path.join(self.save_path, 'text.pkl'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.save_path, 'vocab.pkl'), 'wb'))
        # dump topics as .txt file
        with open(os.path.join(self.save_path, 'topics.txt'), 'w') as f:
            for topic in topic_words:
                f.write(f"{' '.join(topic)}\n")
            # write evaluation scores
            f.write(f"Coherence scores are: {coherence}, topic diversity: {topic_diversity}")
        logger.info(f"Results saved at {self.save_path}")
