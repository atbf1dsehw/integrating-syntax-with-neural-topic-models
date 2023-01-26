import torch
import logging
import numpy as np
import gensim.downloader as gensim_api
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class GetDatasetSeq(Dataset):
    """Get dataset."""

    def __init__(self, seq, seq_target):
        self.seq = seq
        self.seq_target = seq_target

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {'seq': self.seq[idx],
                'seq_target': self.seq_target[idx]}


class GetDatasetBOW(Dataset):
    """Get dataset."""

    def __init__(self, labels, bow):
        self.labels = labels
        self.bow = bow

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        return {'labels': self.labels[idx],
                'bow': self.bow[idx]}


def get_dataloaders_seq(data: dict, settings: dict):
    train_seq = data["train/seq"][0]
    train_seq_target = data["train/seq"][1]
    val_seq = data["val/seq"][0]
    val_seq_target = data["val/seq"][1]
    test_seq = data["test/seq"][0]
    test_seq_target = data["test/seq"][1]
    batch_size = settings.batch_size_seq
    vocab = data["vocab"]
    # train_seq: (docs, seq_len, context_size)
    # train_seq_target: (docs, seq_len)
    # train_seq = train_seq.reshape(-1, train_seq.shape[2])
    # train_seq_target = train_seq_target.reshape(-1)
    # val_seq = val_seq.reshape(-1, val_seq.shape[2])
    # val_seq_target = val_seq_target.reshape(-1)
    # test_seq = test_seq.reshape(-1, test_seq.shape[2])
    # test_seq_target = test_seq_target.reshape(-1)
    logger.info(f"train_seq: {train_seq.shape} raw")
    # remove sequences whose target is [PAD], if [PAD] is in vocab

    if "[PAD]" in vocab:
        pad_idx = vocab["[PAD]"]
        train_seq = train_seq[train_seq_target != pad_idx]
        train_seq_target = train_seq_target[train_seq_target != pad_idx]
        val_seq = val_seq[val_seq_target != pad_idx]
        val_seq_target = val_seq_target[val_seq_target != pad_idx]
        test_seq = test_seq[test_seq_target != pad_idx]
        test_seq_target = test_seq_target[test_seq_target != pad_idx]
        logger.info(f"train_seq: {train_seq.shape} after removing [PAD]")
    # remove sequences whose target is [UNK]

    if "[UNK]" in vocab:
        unk_idx = vocab['[UNK]']
        train_seq = train_seq[train_seq_target != unk_idx]
        train_seq_target = train_seq_target[train_seq_target != unk_idx]
        val_seq = val_seq[val_seq_target != unk_idx]
        val_seq_target = val_seq_target[val_seq_target != unk_idx]
        test_seq = test_seq[test_seq_target != unk_idx]
        test_seq_target = test_seq_target[test_seq_target != unk_idx]
        logger.info(f"train_seq: {train_seq.shape} after removing [UNK]")

        # remove sequences which have [UNK] in context
        train_unks = ~np.any(train_seq == unk_idx, axis=1)
        val_unks = ~np.any(val_seq == unk_idx, axis=1)
        test_unks = ~np.any(test_seq == unk_idx, axis=1)
        train_seq = train_seq[train_unks]
        train_seq_target = train_seq_target[train_unks]
        val_seq = val_seq[val_unks]
        val_seq_target = val_seq_target[val_unks]
        test_seq = test_seq[test_unks]
        test_seq_target = test_seq_target[test_unks]

        logger.info(f"train_seq: {train_seq.shape} after removing [UNK] in context")

    train_dl = GetDatasetSeq(train_seq,
                             train_seq_target)
    val_dl = GetDatasetSeq(val_seq,
                           val_seq_target)
    test_dl = GetDatasetSeq(test_seq,
                            test_seq_target)
    train_dl = DataLoader(train_dl, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_dl, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dl = DataLoader(test_dl, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_dl, val_dl, test_dl


def get_dataloaders_bow(data: dict, settings: dict):
    train_labels = data["train/labels"]
    train_bow = data["train/bow"]
    val_labels = data["val/labels"]
    val_bow = data["val/bow"]
    test_labels = data["test/labels"]
    test_bow = data["test/bow"]
    batch_size = settings.batch_size
    train_dl = GetDatasetBOW(train_labels,
                             train_bow)
    val_dl = GetDatasetBOW(val_labels,
                           val_bow)
    test_dl = GetDatasetBOW(test_labels,
                            test_bow)
    train_dl = DataLoader(train_dl, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_dl, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dl = DataLoader(test_dl, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_dl, val_dl, test_dl


def get_vocab_embeddings(vocab: dict):
    # download glove.6B.50d.txt from https://nlp.stanford.edu/projects/glove/
    model = gensim_api.load('glove-wiki-gigaword-300')
    embeddings = torch.zeros(len(vocab), 300)
    for i, word in enumerate(vocab):
        if word in model:
            embeddings[i] = torch.from_numpy(model[word].copy())
    return embeddings


if __name__ == '__main__':
    vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    embeddings = get_vocab_embeddings(vocab)
