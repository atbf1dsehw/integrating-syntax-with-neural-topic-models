import os
import torch
import logging
import pickle
from src.models.dvae import DVAE
from src.utils import TopicEval

logger = logging.getLogger(__name__)


class DVAETrainer:
    def __init__(self,
                 num_topics: int,
                 vocab: dict,
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
        self.vocab = vocab
        self.text = text
        self.save_path = save_path
        self.num_topics = num_topics

        self.model = DVAE(num_topics=num_topics,
                          vocab_size=vocab_size)
        self.model.to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.logger = {'train/kld': [],
                       'train/rec': [],
                       'val/kld': [],
                       'val/rec': []}
        self.best_val_loss = 1e10
        self.current_val_loss = 0

    def run(self):
        logger.info(f"Training started.")
        for epoch in range(self.max_epochs):
            self.train_one_epoch()
            logger.info(f'Training epoch {epoch} completed; training loss: {self.logger["train/kld"][-1] + self.logger["train/rec"][-1]}')
            self.test_one_epoch()
            logger.info(f'Validation epoch {epoch} completed; validation loss: {self.logger["val/kld"][-1] + self.logger["val/rec"][-1]}')
            if self.current_val_loss < self.best_val_loss:
                self.save_model()
                logger.info(f'Validation loss improved from {self.best_val_loss} to {self.current_val_loss} at epoch {epoch}')
                self.best_val_loss = self.current_val_loss
        logger.info(f"Training completed. Now going for evaluation.")
        # load best model
        self.load_model()
        beta = self.model.decoder.weight.cpu().detach().numpy().T
        # beta is of shape (total_topics, vocab_size)
        eval = TopicEval(vocab=self.vocab, text=self.text)
        topics = eval.get_topics(top_n=10, beta=beta)
        coherence = eval.topic_coherence(metric='c_v', topics=topics)
        topic_diversity = eval.topic_diversity(topics=topics)
        logger.info(f"Evaluation completed. Coherence scores are: {coherence}, topic diversity: {topic_diversity}")
        self.logger['topics'] = topics
        self.logger['tq'] = {'coherence': coherence,
                             'topic_diversity': topic_diversity}
        pickle.dump(self.logger, open(os.path.join(self.save_path, 'logger.pkl'), 'wb'))
        pickle.dump(self.text, open(os.path.join(self.save_path, 'text.pkl'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.save_path, 'vocab.pkl'), 'wb'))
        # dump topics as .txt file
        with open(os.path.join(self.save_path, 'topics.txt'), 'w') as f:
            for topic in topics:
                f.write(f"{' '.join(topic)}\n")
            # write evaluation scores
            f.write(f"Coherence scores are: coherence: {coherence}, topic diversity: {topic_diversity}")
        logger.info(f"Results saved at {self.save_path}")

    def train_one_epoch(self):
        kl_loss = 0
        rec_loss = 0
        for batch in self.train_dl:
            x = batch['bow'].float().to(self.device)
            self.optim.zero_grad()
            x_recon, dist = self.model(x)
            prior_alpha = torch.ones_like(dist.concentration).to(self.device) * 0.02
            prior = torch.distributions.Dirichlet(prior_alpha)
            kl = torch.distributions.kl_divergence(dist, prior)
            kl_loss_curr = 2 * kl.mean()
            rec_loss_curr = -torch.sum(x * x_recon, dim=1).mean()
            loss = kl_loss_curr + rec_loss_curr
            loss.backward()
            self.optim.step()
            kl_loss += kl_loss_curr
            rec_loss += rec_loss_curr
        self.logger['train/kld'].append(kl_loss.item() / len(self.train_dl))
        self.logger['train/rec'].append(rec_loss.item() / len(self.train_dl))

    def test_one_epoch(self):
        kl_loss = 0
        rec_loss = 0
        for batch in self.val_dl:
            x = batch['bow'].float().to(self.device)
            x_recon, dist = self.model(x)
            prior_alpha = torch.ones_like(dist.concentration).to(self.device) * 0.02
            prior = torch.distributions.Dirichlet(prior_alpha)
            kl = torch.distributions.kl_divergence(dist, prior)
            kl_loss_curr = 2 * kl.mean()
            rec_loss_curr = -torch.sum(x * x_recon, dim=1).mean()
            kl_loss += kl_loss_curr
            rec_loss += rec_loss_curr
        self.logger['val/kld'].append(kl_loss.item() / len(self.val_dl))
        self.logger['val/rec'].append(rec_loss.item() / len(self.val_dl))
        self.current_val_loss = (kl_loss.item() + rec_loss.item()) / len(self.val_dl)

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'tm.pt'))

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'tm.pt')))
