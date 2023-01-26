import os
import torch
import logging
import pickle
import numpy as np
from src.models.lm import LM
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LMTrainer:
    def __init__(self,
                 num_topics: int,
                 vocab: dict,
                 vocab_embeddings: torch.Tensor,
                 context_type: str,
                 context_size: int,
                 train_dl: torch.utils.data.DataLoader,
                 val_dl: torch.utils.data.DataLoader,
                 test_dl: torch.utils.data.DataLoader,
                 max_epochs: int,
                 device: str,
                 text: list,
                 lr: float,
                 save_path: str) -> None:
        self.max_epochs = max_epochs
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.device = device
        vocab_size = len(vocab)
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.vocab_embeddings = vocab_embeddings
        self.text = text
        self.save_path = save_path
        self.num_topics = num_topics
        self.context_type = context_type
        self.context_size = context_size
        if '[PAD]' in vocab:
            self.pad_idx = vocab['[PAD]']
        else:
            self.pad_idx = vocab_size + 1

        if self.context_type == 'symmetric':
            input_size = 2 * self.context_size * vocab_embeddings.shape[1]
        else:
            input_size = self.context_size * vocab_embeddings.shape[1]

        self.model = LM(input_dim=input_size,
                        output_dim=vocab_size,
                        vocab_embeddings=vocab_embeddings)
        self.model.to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.logger = {'train/ce_loss': [],
                       'val/ce_loss': []}
        self.best_val_loss = 1e10
        self.current_val_loss = 0

    def run(self):
        logger.info(f"Training started.")
        for epoch in range(self.max_epochs):
            self.model.train()
            self.train_one_epoch()
            logger.info(f'Training epoch {epoch} completed; training loss: {self.logger["train/ce_loss"][-1]}')
            self.model.eval()
            self.test_one_epoch()
            logger.info(f'Validation epoch {epoch} completed; validation loss: {self.logger["val/ce_loss"][-1]}')
            if self.current_val_loss < self.best_val_loss:
                self.save_model()
                logger.info(
                    f'Validation loss improved from {self.best_val_loss} to {self.current_val_loss} at epoch {epoch}')
                self.best_val_loss = self.current_val_loss
        logger.info(f"Training completed. Now going for evaluation.")
        # load best model
        self.load_model()
        # decision module threshold p
        threshold_p = [0.1, 0.3, 0.5]
        top_k = [1, 3, 5]
        for p in threshold_p:
            content_vector, syntax_vector, content_words, syntax_words = self.decision_module_p(p=p)
            with open(os.path.join(self.save_path, f'syntax_words_threshold_{p}.txt'), 'w') as f:
                for word in syntax_words:
                    f.write(word + "\n")
            # content_words to .txt file
            with open(os.path.join(self.save_path, f'content_words_threshold_{p}.txt'), 'w') as f:
                for word in content_words:
                    f.write(word + "\n")
            pickle.dump(content_vector, open(os.path.join(self.save_path, f'content_vector_threshold_{p}.pkl'), 'wb'))
            pickle.dump(syntax_vector, open(os.path.join(self.save_path, f'syntax_vector_threshold_{p}.pkl'), 'wb'))
            pickle.dump(content_words, open(os.path.join(self.save_path, f'content_words_threshold_{p}.pkl'), 'wb'))
            pickle.dump(syntax_words, open(os.path.join(self.save_path, f'syntax_words_threshold_{p}.pkl'), 'wb'))
            
        for k in top_k:
            content_vector, syntax_vector, content_words, syntax_words = self.decision_module_top(n=k)
            with open(os.path.join(self.save_path, f'syntax_words_top_{k}.txt'), 'w') as f:
                for word in syntax_words:
                    f.write(word + "\n")
            # content_words to .txt file
            with open(os.path.join(self.save_path, f'content_words_top_{k}.txt'), 'w') as f:
                for word in content_words:
                    f.write(word + "\n")
            pickle.dump(content_vector, open(os.path.join(self.save_path, f'content_vector_top_{k}.pkl'), 'wb'))
            pickle.dump(syntax_vector, open(os.path.join(self.save_path, f'syntax_vector_top_{k}.pkl'), 'wb'))
            pickle.dump(content_words, open(os.path.join(self.save_path, f'content_words_top_{k}.pkl'), 'wb'))
            pickle.dump(syntax_words, open(os.path.join(self.save_path, f'syntax_words_top_{k}.pkl'), 'wb'))
            
        # content_vector, syntax_vector, content_words, syntax_words = self.syntax_content()
        # # syntax_words to .txt file
        # with open(os.path.join(self.save_path, 'syntax_words.txt'), 'w') as f:
        #     for word in syntax_words:
        #         f.write(word + "\n")
        # # content_words to .txt file
        # with open(os.path.join(self.save_path, 'content_words.txt'), 'w') as f:
        #     for word in content_words:
        #         f.write(word + "\n")
        # pickle.dump(content_vector, open(os.path.join(self.save_path, 'content_vector.pkl'), 'wb'))
        # pickle.dump(syntax_vector, open(os.path.join(self.save_path, 'syntax_vector.pkl'), 'wb'))
        # pickle.dump(content_words, open(os.path.join(self.save_path, 'content_words.pkl'), 'wb'))
        # pickle.dump(syntax_words, open(os.path.join(self.save_path, 'syntax_words.pkl'), 'wb'))
        
        pickle.dump(self.logger, open(os.path.join(self.save_path, 'logger_lm.pkl'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.save_path, 'vocab_lm.pkl'), 'wb'))
        logger.info(f"Evaluation completed.")
        return syntax_vector, content_vector

    def train_one_epoch(self):
        ce_loss = 0
        for batch in self.train_dl:
            x = batch['seq'].to(self.device)
            # x is of shape (batch_size, context_size)
            y = batch['seq_target'].to(self.device)
            # y is of shape (batch_size)
            self.optim.zero_grad()
            y_pred = self.model(x)
            # y_pred is of shape (batch_size, vocab_size)
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            self.optim.step()
            ce_loss += loss.item()
        self.logger['train/ce_loss'].append(ce_loss / len(self.train_dl))

    def test_one_epoch(self):
        ce_loss = 0
        with torch.no_grad():
            for batch in self.val_dl:
                x = batch['seq'].to(self.device)
                # x is of shape (batch_size, context_size)
                y = batch['seq_target'].to(self.device)
                # y is of shape (batch_size)
                y_pred = self.model(x)
                # y_pred is of shape (batch_size, vocab_size)
                loss = F.cross_entropy(y_pred, y)
                ce_loss += loss.item()
        self.logger['val/ce_loss'].append(ce_loss / len(self.val_dl))
        self.current_val_loss = ce_loss / len(self.val_dl)

    def decision_module_top(self, n):
        dec_vector = torch.zeros(len(self.vocab))
        idx2word = {v: k for k, v in self.vocab.items()}
        with torch.no_grad():
            for idx, batch in enumerate(self.train_dl):
                x = batch['seq'].to(self.device)
                # x is of shape (batch_size, context_size)
                y = batch['seq_target']
                # y is of shape (batch_size)
                y_pred = self.model(x)
                # y_pred is of shape (batch_size, vocab_size)
                top_3 = torch.topk(y_pred, n, dim=-1).indices
                # top_3 is of shape (batch_size, 3)
                top_3 = top_3.cpu().numpy()
                y = y.cpu().numpy()
                for i in range(x.shape[0]):
                    if y[i] != self.pad_idx:
                        if y[i] in top_3[i]:
                            # syntax
                            dec_vector[y[i]] += 1
                        else:
                            # content
                            dec_vector[y[i]] -= 1
        content_vector = dec_vector < 0
        syntax_vector = dec_vector > 0
        syntax_words = [idx2word[i] for i in range(len(self.vocab)) if syntax_vector[i]]
        content_words = [idx2word[i] for i in range(len(self.vocab)) if content_vector[i]]
        return content_vector, syntax_vector, content_words, syntax_words
    
    def decision_module_p(self, p):
        dec_vector = torch.zeros(len(self.vocab))
        idx2word = {v: k for k, v in self.vocab.items()}
        with torch.no_grad():
            for idx, batch in enumerate(self.train_dl):
                x = batch['seq'].to(self.device)
                # x is of shape (batch_size, context_size)
                y = batch['seq_target']
                # y is of shape (batch_size)
                y_pred = self.model(x)
                y_pred = F.softmax(y_pred, dim=-1)
                # y_pred is of shape (batch_size, vocab_size)
                y_pred = y_pred.cpu().numpy()
                y_pred = y_pred > p
                y = y.cpu().numpy()
                for i in range(x.shape[0]):
                    if y[i] != self.pad_idx:
                        if y_pred[i, y[i]]:
                            # syntax
                            dec_vector[y[i]] += 1
                        else:
                            # content
                            dec_vector[y[i]] -= 1
        content_vector = dec_vector < 0
        syntax_vector = dec_vector > 0
        syntax_words = [idx2word[i] for i in range(len(self.vocab)) if syntax_vector[i]]
        content_words = [idx2word[i] for i in range(len(self.vocab)) if content_vector[i]]
        return content_vector, syntax_vector, content_words, syntax_words

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'lm.pt'))

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'lm.pt')))
