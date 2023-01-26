import os
import pickle
import string
import logging
import numpy as np
import spacy
from src.data.tokenize import tokenize_and_clean
from src.data.save_data import SaveData
from src.data.context import make_context_vector
from sklearn.feature_extraction.text import CountVectorizer  # for bag of words

nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)


def read_and_clean_data(data_name: str,
                        max_doc_len: int,
                        preprocess: bool,
                        return_seq: bool,
                        context_type: str,
                        context_size: int,
                        total_train_docs=None,
                        total_val_docs=None):
    """ This function reads the data and cleans it.
    This is the main function to read the data.
    It is the only function that should be called from outside from this package.
    It reads the data from the raw_data folder and cleans it.
    Args:
        data_name (str): Name of the dataset.
        max_doc_len (int): Maximum length of the document.
        preprocess (bool): Whether to preprocess the data or not.
        return_seq (list): Do we want to return the sequences or not.
        context_type (str): Type of context to use. (symmetric or asymmetric)
        context_size (int): Size of the context. (number of words to use as context)
        total_train_docs (_type_, optional): Defaults to None.
        total_val_docs (_type_, optional): Defaults to None.

    Returns:
        data (dict): Dictionary containing the data.
        data = {
            'train/seq': list of context vectors and target words for training (context_vec, context_target),
            'train/labels': list of labels for training,
            'train/bow': list of bag of words for training,
            'val/seq': list of context vectors and target words for validation (context_vec, context_target),
            'val/labels': list of labels for validation,
            'val/bow': list of bag of words for validation,
            'test/seq': list of context vectors and target words for testing (context_vec, context_target),
            'test/labels': list of labels for testing,
            'test/bow': list of bag of words for testing,
            'vocab': list of words in the vocabulary,
            'text': list of documents for reference
        }
            
    """
    if return_seq and preprocess:
        logger.info(f'Preprocessing is True, so we will not return sequences.')
        raise ValueError('Preprocessing is True, so we will not return sequences.')
    # Check if the data is already saved or not.
    if not os.path.exists('./src/datasets/raw_data/' + data_name + '.pkl'):
        logger.info(f'No data found for {data_name}! Generating and saving data...')
        SaveData(data_name=data_name)
        logger.info(f'Data saved for {data_name}!')
    # Read the data
    raw_data = pickle.load(open('./src/datasets/raw_data/' + data_name + '.pkl', 'rb'))
    data = {}
    logger.info(f'Start encoding data...')
    if total_train_docs is not None:
        logger.info(f'Since total_train_docs is not None, we will only use {total_train_docs} train docs.')
        for key in raw_data.keys():
            if key[:5] == 'train':
                raw_data[key] = raw_data[key][:total_train_docs]
                logger.info(f'Total train docs: {len(raw_data[key])}')

    if total_val_docs is not None:
        logger.info(f'Since total_val_docs is not None, we will only use {total_val_docs} val/test docs.')
        for key in raw_data.keys():
            if key[:3] == 'val' or key[:4] == 'test':
                raw_data[key] = raw_data[key][:total_val_docs]
                logger.info(f'Total val/test docs: {len(raw_data[key])}')

    for key in raw_data.keys():
        # go through all the keys in the raw_data dictionary (train/text, train/label, val/text, val/label, test/text, test/label)
        if key[-4:] == 'text':
            # if the key is a text key, then we need to tokenize and clean the data
            # text is a list of documents (list of strings)
            data[key] = tokenize_and_clean(raw_data[key], max_doc_len)
            if key == 'train/text':
                # if the key is train/text, then we need to save it to use as reference corpus.
                # we remove the [PAD] tokens from the text
                # we also create a vocab list out of the tokens in the text.
                text = []
                vocab = []
                for i in data[key]:
                    a_doc = []
                    for j in i:
                        if j != '[PAD]':
                            a_doc.append(j)
                        if j not in vocab:
                            vocab.append(j)
                    text.append(a_doc)
                data['text'] = text
                avg_doc_len = np.mean([len(i) for i in text])
                print(f'Average document length: {avg_doc_len}')
                logger.info(f'Average document length: {avg_doc_len}')
            # we need to join the tokens in the text with '||-sep-||' to create a string.
            # this is because CountVectorizer can only take a list of strings as input.
            data[key] = ['||-sep-||'.join(i) for i in data[key]]
        else:
            # if the key is not a text key, then we just copy the data from raw_data to data.
            data[key] = raw_data[key]
    logger.info('Data encoded!')
    special_tokens = ['[PAD]', '[BOD]', '[EOD]', '[NUM]', '[ALNUM]', '[URL/EMAIL]', '[UNK]']
    for i in special_tokens:
        if i not in vocab:
            vocab.append(i)
    if preprocess:
        spacy_stops = list(nlp.Defaults.stop_words)
        stopwords = spacy_stops + ['[PAD]', '[BOD]', '[EOD]', '[NUM]', '[ALNUM]', '[URL/EMAIL]'] + list(
            string.punctuation) + list(string.ascii_lowercase) + ["'nt", "'t", "'ve"]
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split('||-sep-||'), max_df=0.70, min_df=0.02, lowercase=False,
                                     stop_words=stopwords, token_pattern=None)
    else:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split('||-sep-||'), lowercase=False, token_pattern=None,
                                     min_df=2)
        # add the special tokens to the vocab
    data['train/bow'] = vectorizer.fit(data['train/text'])
    if not preprocess:
        if '[UNK]' not in vectorizer.vocabulary_:
            vectorizer.vocabulary_['[UNK]'] = len(vectorizer.vocabulary_)
    data['train/bow'] = vectorizer.transform(data['train/text']).toarray()
    data['val/bow'] = vectorizer.transform(data['val/text']).toarray()
    data['test/bow'] = vectorizer.transform(data['test/text']).toarray()
    vocab = vectorizer.vocabulary_
    data['vocab'] = vocab
    # if preprocess:
    for i in special_tokens:
        if i in vocab:
            data['train/bow'][:, vocab[i]] = 0
            data['val/bow'][:, vocab[i]] = 0
            data['test/bow'][:, vocab[i]] = 0
    logger.info(f"Vocab size: {len(vocab)}")
    if return_seq:
        data['train/seq'] = []
        data['val/seq'] = []
        data['test/seq'] = []
        for key in data.keys():
            if key[-4:] == 'text' and key != 'text':
                new_key = key[:-4] + 'seq'
                new_data = []
                for i in data[key]:
                    doc_idx = []
                    doc_text = i.split('||-sep-||')
                    for j in doc_text:
                        if j in vocab:
                            doc_idx.append(vocab[j])
                        else:
                            doc_idx.append(vocab['[UNK]'])
                    new_data.append(doc_idx)
                context, target = make_context_vector(context_type, context_size, new_data)
                data[new_key] = (np.array(context), np.array(target))
    data['train/labels'] = np.array(data['train/labels'])
    data['val/labels'] = np.array(data['val/labels'])
    data['test/labels'] = np.array(data['test/labels'])
    data.pop('train/text')
    data.pop('val/text')
    data.pop('test/text')
    return data


if __name__ == '__main__':
    datasets = ["20ng", "yelp", "imdb", "ag_news", "rotten_tomatoes", "amazon_polarity", "govreport-summarization"]
    for data_name in datasets:
        data = read_and_clean_data(data_name=data_name, preprocess=False, return_seq=True, max_doc_len=-1,
                                   context_type='symmetric', context_size=2)
        vocab = data['vocab']
        # vocab to list
        vocab_list = []
        for key in vocab.keys():
            vocab_list.append(key)
        # sort the vocab list
        vocab_list.sort()
        print(f"Dataset: {data_name}")
        print(f"Vocab size: {len(vocab_list)}")
        print(f"Vocab: {vocab_list[0:1000]}")
        # print(data['train/seq'][0][0], data['train/seq'][1][0])
        # print(data.keys())
        break
