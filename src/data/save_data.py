import os
import random
import pickle
import logging
import numpy as np
from sklearn.datasets import fetch_20newsgroups  # for 20 newsgroups dataset
from datasets import load_dataset  # for all the huggingface datasets

logger = logging.getLogger(__name__)


class SaveData:
    """This class is used to save the data in a pickle file.
    Parameters:
    -----------
    data_name: str
        Name of the dataset.
    
    Saves the data in a pickle file.
    Data is saved in the following format:
    data = {
        'train/text': list of training documents, (list of strings)
        'train/labels': list of training labels, (list of ints)
        'val/text': list of validation documents, (list of strings)
        'val/labels': list of validation labels, (list of ints)
        'test/text': list of test documents, (list of strings)
        'test/labels': list of test labels, (list of ints)
    }
    Data is saved in the following path: ./src/datasets/data/{data_name}.pkl
    If you want to save the data in a different path, change the path in the save_data function.
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

        elif self.data_name == 'govreport-summarization':
            data_train = load_dataset('ccdv/govreport-summarization', split='train')
            data_val = load_dataset('ccdv/govreport-summarization', split='validation')
            data_test = load_dataset('ccdv/govreport-summarization', split='test')
            data_train = data_train.shuffle(seed=42)
            data_test = data_test.shuffle(seed=42)
            data_val = data_val.shuffle(seed=42)
            data['train/text'], data['train/labels'] = data_train['summary'], [0 for i in
                                                                               range(len(data_train['summary']))]
            data['val/text'], data['val/labels'] = data_val['summary'], [0 for i in range(len(data_val['summary']))]
            data['test/text'], data['test/labels'] = data_test['summary'], [0 for i in
                                                                            range(len(data_train['summary']))]

        # make sure all the values are stored in lists
        logger.info(f"Dataset downloaded and splits are generated. Converting {self.data_name} dataset to list...")
        # if data more than 50000, then take only 50000 samples
        if len(data['train/text']) > 50000:
            data['train/text'] = data['train/text'][:50000]
            data['train/labels'] = data['train/labels'][:50000]
        if len(data['val/text']) > 10000:
            data['val/text'] = data['val/text'][:10000]
            data['val/labels'] = data['val/labels'][:10000]
        if len(data['test/text']) > 10000:
            data['test/text'] = data['test/text'][:10000]
            data['test/labels'] = data['test/labels'][:10000]
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
