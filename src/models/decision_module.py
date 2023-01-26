import torch
from src.models.lm import LM
import torch.nn.functional as F


def load_lm(path: str,
            input_dim: int,
            output_dim: int,
            vocab_embeddings: torch.Tensor) -> LM:
    lm = LM(input_dim=input_dim, output_dim=output_dim, vocab_embeddings=vocab_embeddings)
    lm.load_state_dict(torch.load(path))
    return lm


def top_n_words(vocab: dict,
                n: int,
                data: torch.utils.data.DataLoader,
                device: str,
                model: LM):
    dec_vector = torch.zeros(len(vocab))
    idx2word = {v: k for k, v in vocab.items()}
    with torch.no_grad():
        for idx, batch in enumerate(data):
            x = batch['seq'].to(device)
            # x is of shape (batch_size, context_size)
            y = batch['seq_target']
            # y is of shape (batch_size)
            y_pred = model(x)
            # y_pred is of shape (batch_size, vocab_size)
            top_n = torch.topk(y_pred, n, dim=-1).indices
            # top_n is of shape (batch_size, n)
            top_n = top_n.cpu().numpy()
            y = y.cpu().numpy()
            for i in range(x.shape[0]):
                if y[i] in top_n[i]:
                    # syntax
                    dec_vector[y[i]] += 1
                else:
                    # content
                    dec_vector[y[i]] -= 1
    content_vector = dec_vector < 0
    syntax_vector = dec_vector > 0
    syntax_words = [idx2word[i] for i in range(len(vocab)) if syntax_vector[i]]
    content_words = [idx2word[i] for i in range(len(vocab)) if content_vector[i]]
    print(len(syntax_words), len(content_words))
    return content_vector, syntax_vector, content_words, syntax_words


def probability_threshold(vocab: dict,
                          p: float,
                          data: torch.utils.data.DataLoader,
                          device: str,
                          model: LM):
    dec_vector = torch.zeros(len(vocab))
    idx2word = {v: k for k, v in vocab.items()}
    with torch.no_grad():
        for idx, batch in enumerate(data):
            x = batch['seq'].to(device)
            # x is of shape (batch_size, context_size)
            y = batch['seq_target']
            # y is of shape (batch_size)
            y_pred = model(x)
            y_pred = F.softmax(y_pred, dim=-1)
            # check which indices have a probability higher than p
            y_pred = y_pred.cpu().numpy()
            y_pred = y_pred > p
            y = y.cpu().numpy()
            for i in range(x.shape[0]):
                if y_pred[i, y[i]]:
                    # syntax
                    dec_vector[y[i]] += 1
                else:
                    # content
                    dec_vector[y[i]] -= 1
    content_vector = dec_vector < 0
    syntax_vector = dec_vector > 0
    syntax_words = [idx2word[i] for i in range(len(vocab)) if syntax_vector[i]]
    content_words = [idx2word[i] for i in range(len(vocab)) if content_vector[i]]
    return content_vector, syntax_vector, content_words, syntax_words


def distinguish_words(vocab: dict,
                      data: torch.utils.data.DataLoader,
                      vocab_embeddings: torch.Tensor,
                      device: str,
                      path: str,
                      lm_input_size: int,
                      n: int = None,
                      p: float = None,
                      dec_type: str = 'top_n'):
    model = load_lm(path, input_dim=lm_input_size, output_dim=len(vocab), vocab_embeddings=vocab_embeddings)
    model.to(device)
    model.eval()
    if dec_type == 'top_n':
        return top_n_words(vocab, n, data, device, model)
    elif dec_type == 'probability_threshold':
        return probability_threshold(vocab, p, data, device, model)
    else:
        raise ValueError(f'Unknown decision type: {dec_type}')
