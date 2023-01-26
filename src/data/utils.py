import os
import random
import pickle
import string
import logging
import numpy as np
from sklearn.datasets import fetch_20newsgroups  # for 20 newsgroups dataset
from sklearn.feature_extraction.text import CountVectorizer  # for bag of words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # for stopwords
from datasets import load_dataset  # for all the huggingface datasets
from tokenizers import Tokenizer  # for tokenizing the text
from tokenizers import normalizers  # for normalizing the text
from tokenizers import pre_tokenizers  # for pre-tokenizing the text
from tokenizers import models  # for building the tokenizer model
from tokenizers import trainers  # for training the tokenizer model

logger = logging.getLogger(__name__)


class SaveData:
    """This class is used to save the data in a pickle file.
    Parameters:
    -----------
    data_name: str
        Name of the dataset.
    """

    def __init__(self,
                 data_name: str):
        self.data_name = data_name
        logger.info(f"Downloading {self.data_name} dataset...")
        self.raw_data = self.download_data()
        logger.info(f"Saving {self.data_name} dataset...")
        self.save_data()
        logger.info(f"Downloaded and saved {self.data_name} dataset")

    def download_data(self) -> dict:
        """This function is used to download the dataset.
        Returns:
        --------
        data: dict
            Dictionary containing the train, validation and test data.
        """
        logger.info(f"Downloading {self.data_name} dataset...")
        data = {}
        if self.data_name == "20ng":
            data_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True,
                                            random_state=42)
            data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True,
                                           random_state=42)
            data['train/text'], data['train/labels'], data['test/text'], data[
                'test/labels'] = data_train.data, data_train.target, data_test.data, data_test.target
            data['train/text'], data['train/labels'], data['val/text'], data['val/labels'] = self.train_val_split(
                data['train/text'], data['train/labels'])

        elif self.data_name == 'yelp':
            data_train = load_dataset('yelp_review_full', split='train')
            data_test = load_dataset('yelp_review_full', split='test')
            data_train = data_train.shuffle(seed=42)
            data_test = data_test.shuffle(seed=42)
            data['train/text'], data['train/labels'] = data_train['text'], data_train['label']
            data['test/text'], data['test/labels'] = data_test['text'], data_test['label']
            data['train/text'], data['train/labels'], data['val/text'], data['val/labels'] = self.train_val_split(
                data['train/text'], data['train/labels'])

        elif self.data_name == 'imdb':
            data_train = load_dataset('imdb', split='train')
            data_test = load_dataset('imdb', split='test')
            data_train = data_train.shuffle(seed=42)
            data_test = data_test.shuffle(seed=42)
            data['train/text'], data['train/labels'] = data_train['text'], data_train['label']
            data['test/text'], data['test/labels'] = data_test['text'], data_test['label']
            data['train/text'], data['train/labels'], data['val/text'], data['val/labels'] = self.train_val_split(
                data['train/text'], data['train/labels'])

        elif self.data_name == 'ag_news':
            data_train = load_dataset('ag_news', split='train')
            data_test = load_dataset('ag_news', split='test')
            data_train = data_train.shuffle(seed=42)
            data_test = data_test.shuffle(seed=42)
            data['train/text'], data['train/labels'] = data_train['text'], data_train['label']
            data['test/text'], data['test/labels'] = data_test['text'], data_test['label']
            data['train/text'], data['train/labels'], data['val/text'], data['val/labels'] = self.train_val_split(
                data['train/text'], data['train/labels'])

        elif self.data_name == 'rotten_tomatoes':
            data_train = load_dataset('rotten_tomatoes', split='train')
            data_val = load_dataset('rotten_tomatoes', split='validation')
            data_test = load_dataset('rotten_tomatoes', split='test')
            data_train = data_train.shuffle(seed=42)
            data_val = data_val.shuffle(seed=42)
            data_test = data_test.shuffle(seed=42)
            data['train/text'], data['train/labels'] = data_train['text'], data_train['label']
            data['val/text'], data['val/labels'] = data_val['text'], data_val['label']
            data['test/text'], data['test/labels'] = data_test['text'], data_test['label']

        elif self.data_name == 'amazon_polarity':
            data_train = load_dataset('amazon_polarity', split='train')
            data_test = load_dataset('amazon_polarity', split='test')
            data_train = data_train.shuffle(seed=42)
            data_test = data_test.shuffle(seed=42)
            data['train/text'], data['train/labels'] = data_train['content'], data_train['label']
            data['test/text'], data['test/labels'] = data_test['content'], data_test['label']
            data['train/text'], data['train/labels'], data['val/text'], data['val/labels'] = self.train_val_split(
                data['train/text'], data['train/labels'])
        # make sure all the values are stored in lists
        logger.info(f"Dataset downloaded and splits are generated. Converting {self.data_name} dataset to list...")
        # if data more than 50000, limiting to 50K documents, use all if needed (takes more time to train)
        if len(data['train/text']) > 50000:
            data['train/text'] = data['train/text'][:50000]
            data['train/labels'] = data['train/labels'][:50000]
        if len(data['val/text']) > 10000:
            data['val/text'] = data['val/text'][:10000]
            data['val/labels'] = data['val/labels'][:10000]
        if len(data['test/text']) > 10000:
            data['test/text'] = data['test/text'][:10000]
            data['test/labels'] = data['test/labels'][:10000]
        # convert to list
        for key in data.keys():
            if not isinstance(data[key], list):
                data[key] = data[key].tolist()
        return data

    def train_val_split(self, train_text, train_labels):
        """ This function splits the training data into training and validation data.
        Args:
            train_text: list of training text
            train_labels: list of training labels
            
        Returns:
            train_text_new: list of training text after splitting
            train_labels_new: list of training labels after splitting
            val_text: list of validation text
            val_labels: list of validation labels
        """
        train_text, train_labels = np.array(train_text), np.array(train_labels)
        train_data = range(len(train_text))
        val_data = random.sample(train_data, int(len(train_data) * 0.1))
        train_data = list(set(train_data) - set(val_data))
        train_text_new, train_labels_new = train_text[train_data], train_labels[train_data]
        val_text, val_labels = train_text[val_data], train_labels[val_data]
        train_text_new, train_labels_new = train_text_new.tolist(), train_labels_new.tolist()
        val_text, val_labels = val_text.tolist(), val_labels.tolist()
        return train_text_new, train_labels_new, val_text, val_labels

    def save_data(self):
        """ This function saves the data in a pickle file.

        Returns:
            None
        """
        # save the data in /src/datasets/raw_data (create a folder if it doesn't exist)
        logger.info(f"Saving {self.data_name} dataset in pickle file...")
        if not os.path.exists('./src/datasets/raw_data'):
            os.makedirs('./src/datasets/raw_data')
        pickle.dump(self.raw_data, open('./src/datasets/raw_data/' + self.data_name + '.pkl', 'wb'))


class SaveTokenizer:
    """ This class saves the tokenizer in a pickle file.
    Args:
        data_name: name of the dataset.
    """

    def __init__(self,
                 data_name: str):
        self.data_name = data_name
        logger.info(f'Loading data...')
        self.raw_data = pickle.load(open('./src/datasets/raw_data/' + self.data_name + '.pkl', 'rb'))
        logger.info(f'Data loaded!, now tokenizing...')
        self.all_text = self.raw_data['train/text'] + self.raw_data['val/text'] + self.raw_data['test/text']
        self.tokenize()

    def tokenize(self):
        """ This function tokenizes the data."""
        logger.info(f"Tokenizing {self.data_name} dataset...")
        normalize = normalizers.Sequence([normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True,
                                                                     strip_accents=True, lowercase=True)])
        pre_tokenize = pre_tokenizers.Whitespace()
        model = models.WordLevel(unk_token="[UNK]")
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[BOD]", "[EOD]", "[PAD]"])
        tokenizer = Tokenizer(model=model)
        tokenizer.normalizer = normalize
        tokenizer.pre_tokenizer = pre_tokenize
        tokenizer.train_from_iterator(self.all_text, trainer=trainer)
        if not os.path.exists('./src/datasets/trained_tokenizers'):
            os.makedirs('./src/datasets/trained_tokenizers')
        tokenizer.save('./src/datasets/trained_tokenizers/' + self.data_name + '.json')
        logger.info(f"Tokenizer saved in ./src/datasets/trained_tokenizers/{self.data_name}.json")


def read_and_clean_data(data_name: str,
                        max_doc_len: int,
                        preprocess: bool,
                        total_train_docs=None,
                        total_val_docs=None):
    """ This function reads the data and cleans it.
    Args:
        data_name (str): Name of the dataset.
        max_doc_len (int): Maximum length of the document.
        preprocess (bool): Whether to preprocess the data or not.
        total_train_docs (_type_, optional): Defaults to None.
        total_val_docs (_type_, optional): Defaults to None.

    Returns:
        data (dict): Dictionary containing the data.
    """
    raw_data = pickle.load(open('./src/datasets/raw_data/' + data_name + '.pkl', 'rb'))
    tokenizer = Tokenizer.from_file('./src/datasets/trained_tokenizers/' + data_name + '.json')
    vocab = tokenizer.get_vocab()
    data = {}
    logger.info(f'Start encoding data...')
    if total_train_docs is not None:
        for key in raw_data.keys():
            if key[:5] == 'train':
                raw_data[key] = raw_data[key][:total_train_docs]
                logger.info(f'Total train docs: {len(raw_data[key])}')

    if total_val_docs is not None:
        for key in raw_data.keys():
            if key[:3] == 'val' or key[:4] == 'test':
                raw_data[key] = raw_data[key][:total_val_docs]
                logger.info(f'Total val/test docs: {len(raw_data[key])}')

    for key in raw_data.keys():
        if key[-4:] == 'text':
            data[key] = tokenizer.encode_batch(raw_data[key])
            data[key] = [i.tokens for i in data[key]]
            data[key] = clean_data(data[key], max_doc_len)
            if key == 'train/text':
                text = []
                for i in data[key]:
                    a_doc = []
                    for j in i:
                        if j != '[PAD]':
                            a_doc.append(j)
                    text.append(a_doc)
                data['text'] = text
            data[key] = ['||-sep-||'.join(i) for i in data[key]]
        else:
            data[key] = raw_data[key]
    logger.info('Data encoded!')
    if preprocess:
        stopwords = list(ENGLISH_STOP_WORDS) + ['[PAD]', '[BOD]', '[EOD]', '[NUM]', '[ALNUM]', '[URL/EMAIL]'] + list(
            string.punctuation) + list(string.ascii_lowercase) + ["'nt", "'t", "'ve"]
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split('||-sep-||'), max_df=0.90, min_df=20, lowercase=False,
                                     stop_words=stopwords)
    else:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split('||-sep-||'), vocabulary=vocab, lowercase=False,
                                     min_df=3)
    data['train/bow'] = vectorizer.fit_transform(data['train/text']).toarray()
    data['val/bow'] = vectorizer.transform(data['val/text']).toarray()
    data['test/bow'] = vectorizer.transform(data['test/text']).toarray()
    vocab = vectorizer.vocabulary_
    data['vocab'] = vocab
    if preprocess:
        data['train/bow'][:, vocab['[UNK]']] = 0
        data['val/bow'][:, vocab['[UNK]']] = 0
        data['test/bow'][:, vocab['[UNK]']] = 0
    logger.info(f"Vocab size: {len(vocab)}")
    # if '[UNK]' not in vocab:
    #     vocab['[UNK]'] = len(vocab)
    # token to index
    ## add new key to data
    data['train/seq'] = []
    data['val/seq'] = []
    data['test/seq'] = []
    for key in data.keys():
        if key[-4:] == 'text' and key != 'text':
            new_key = key[:-4] + 'seq'
            for i in data[key]:
                doc_idx = []
                doc_text = i.split('||-sep-||')
                for j in doc_text:
                    if j in vocab:
                        doc_idx.append(vocab[j])
                    else:
                        doc_idx.append(vocab['[UNK]'])
                data[new_key].append(doc_idx)
    data.pop('train/text')
    data.pop('val/text')
    data.pop('test/text')
    return data


def clean_data(data: list, max_doc_len: int):
    """ This function cleans the data."""
    # data is in list of list format
    all_punct = string.punctuation
    clean_data = []
    for idx, i in enumerate(data):
        clean_doc = []
        clean_doc.append('[BOD]')
        last_period_idx = 0
        for j in i:
            if len(j) > 15:
                continue
            if '@' in j or 'http' in j or 'www' in j or '.com' in j:
                clean_doc.append('[URL/EMAIL]')
            elif j.isnumeric():
                clean_doc.append('[NUM]')
            elif j.isalpha():
                clean_doc.append(j)
            elif j.isalnum():
                clean_doc.append('[ALNUM]')
            elif j in all_punct:
                clean_doc.append(j)
            else:
                clean_doc.append('[UNK]')
            if j == '.':
                last_period_idx = len(clean_doc)
            if len(clean_doc) == max_doc_len - 1:
                if last_period_idx > 0:
                    clean_doc = clean_doc[:last_period_idx]
                break
        clean_doc.append('[EOD]')
        if len(clean_doc) < max_doc_len:
            clean_doc = clean_doc + ['[PAD]'] * (max_doc_len - len(clean_doc))
        clean_data.append(clean_doc)
    return clean_data


def make_context_vector(context_type: str,
                        context_size: int,
                        data: dict):
    """ This function adds context vector to the data."""
    if context_type == 'symmetric':
        for key in data.keys():
            if key[-3:] == 'seq':
                context_vec = []
                context_target = []
                for i in data[key]:
                    context_doc = []
                    target_doc = []
                    for j in range(len(i)):
                        if j >= context_size and j < len(i) - context_size:
                            context_doc.append(i[j - context_size:j + context_size + 1])
                            target_doc.append(i[j])
                    context_vec.append(context_doc)
                    context_target.append(target_doc)
                data[key] = [context_vec, context_target]

    elif context_type == 'asymmetric':
        for key in data.keys():
            if key[-3:] == 'seq':
                context_vec = []
                context_target = []
                for i in data[key]:
                    context_doc = []
                    target_doc = []
                    for j in range(len(i)):
                        if j >= context_size and j < len(i) - context_size:
                            context_doc.append(i[j - context_size:j])
                            target_doc.append(i[j])
                    context_vec.append(context_doc)
                    context_target.append(target_doc)
                data[key] = [context_vec, context_target]
    return data


def prepare_data(data_name):
    """ This function prepares the data."""
    SaveData(data_name)
    SaveTokenizer(data_name)


if __name__ == "__main__":
    # all_data_list = ['20ng', 'yelp', 'imdb', 'ag_news', 'rotten_tomatoes', 'amazon_polarity']
    all_data_list = ['20ng', 'ag_news', 'rotten_tomatoes']
    for data_name in all_data_list:
        # prepare_data(data_name)
        data = read_and_clean_data(data_name, max_doc_len=201, preprocess=True)
        print(data['train/seq'][0])
        # logger.info(data['train/text'][0])
        print(data['text'][0])
        print(data.keys())
        print(data['vocab']['[UNK]'])
        print(len(data['vocab']))
        print(f'context vector')
        data = make_context_vector('asymmetric', 2, data)
        print(data['train/seq'][0][0], data['train/seq'][1][0])
        break
