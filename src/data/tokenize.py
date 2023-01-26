import spacy
import string
from sklearn.datasets import fetch_20newsgroups  # for 20 newsgroups dataset

nlp = spacy.load('en_core_web_sm')

tokenizer = nlp.tokenizer


def tokenize_and_clean(data: list,
                       max_doc_len: int):
    """
    This function tokenizes and cleans the data.
    
    Parameters
    ----------
    data : list
        List of documents. Each document is a string.
    max_doc_len : int
        Maximum length of the document.
    
    Returns
    -------
    tokenized_and_cleaned_data : list
        List of tokenized and cleaned documents. Each document is a list of tokens. (list of lists)
    """
    # normalize the text
    data = [doc.lower() for doc in data]
    # now tokenize the text
    tokenized_and_cleaned_data = []
    for doc in tokenizer.pipe(data, batch_size=1000):
        clean_doc = []
        clean_doc.append('[BOD]')
        last_period_idx = 0
        for token in doc:
            if len(token.text) > 10:
                continue
            if len(token.text) < 1:
                continue
            if token.is_space:
                continue
            if "\\n" in token.text:
                continue
            if token.like_url or token.like_email:
                clean_doc.append('[URL/EMAIL]')
            elif token.is_digit:
                clean_doc.append('[NUM]')
            elif token.is_alpha:
                clean_doc.append(token.text)
            elif token.text.isalnum():
                clean_doc.append('[ALNUM]')
            elif token.text in string.punctuation:
                # elif token.is_punct:
                clean_doc.append(token.text)
            elif "." in token.text:
                clean_doc.append(".")
            elif token.is_oov:
                clean_doc.append('[UNK]')
            else:
                clean_doc.append(token.text)
            if token.text == '.':
                last_period_idx = len(clean_doc)
            if len(clean_doc) == max_doc_len - 1 and max_doc_len > 0:
                if last_period_idx > 0:
                    clean_doc = clean_doc[:last_period_idx]
                else:
                    clean_doc = clean_doc[:max_doc_len - 1]
                break
        clean_doc.append('[EOD]')
        if len(clean_doc) < max_doc_len and max_doc_len > 0:
            clean_doc = clean_doc + ['[PAD]'] * (max_doc_len - len(clean_doc))
        tokenized_and_cleaned_data.append(clean_doc)
    return tokenized_and_cleaned_data


if __name__ == '__main__':
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).data
    print(data[-1])
    clean = tokenize_and_clean(data, -1)
    new_docs = []
    vocab = set()
    for doc in clean:
        vocab.update(doc)
        while '[PAD]' in doc:
            doc.remove('[PAD]')
        new_docs.append(' '.join(doc))
    vocab = list(vocab)
    vocab.sort()
    print(len(vocab))
    print(new_docs[0])
    print(vocab[0:50])
