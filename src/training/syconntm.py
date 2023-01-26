import os
import torch
import logging
import pickle
from src.models.syconntm import SyConNTM
from src.utils import TopicEval

logger = logging.getLogger(__name__)


class SyConNTMTrainer:
    def __init__(self,
                 num_topics: int,
                 num_syn_topics: int,
                 vocab: dict,
                 train_dl: torch.utils.data.DataLoader,
                 val_dl: torch.utils.data.DataLoader,
                 test_dl: torch.utils.data.DataLoader,
                 syntax_vector: torch.Tensor,
                 content_vector: torch.Tensor,
                 max_epochs: int,
                 device: str,
                 text: list,
                 lr: float,
                 lambda_: float,
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
        self.num_syn_topics = num_syn_topics
        self.syntax_vector = syntax_vector.to(self.device)
        self.content_vector = content_vector.to(self.device)
        self.lambda_ = lambda_

        self.model = SyConNTM(num_con_topics=num_topics,
                              num_syn_topics=num_syn_topics,
                              vocab_size=vocab_size)
        self.model.to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.logger = {'train/kld_con': [],
                       'train/rec_con': [],
                       'train/kld_syn': [],
                       'train/rec_syn': [],
                       'train/loss': [],
                       'val/kld_con': [],
                       'val/rec_con': [],
                       'val/kld_syn': [],
                       'val/rec_syn': [],
                       'val/loss': []}
        self.best_val_loss = 1e10
        self.current_val_loss = 0

    def run(self):
        logger.info(f"Training started.")
        for epoch in range(self.max_epochs):
            self.train_one_epoch()
            logger.info(f'Training epoch {epoch} completed; training loss: {self.logger["train/loss"][-1]}')
            self.test_one_epoch()
            logger.info(f'Validation epoch {epoch} completed; validation loss: {self.logger["val/loss"][-1]}')
            if self.current_val_loss < self.best_val_loss:
                self.save_model()
                logger.info(
                    f'Validation loss improved from {self.best_val_loss} to {self.current_val_loss} at epoch {epoch}')
                self.best_val_loss = self.current_val_loss
        logger.info(f"Training completed. Now going for evaluation.")
        # load best model
        self.load_model()
        beta_con = self.model.decoder_con.weight.cpu().detach().numpy().T
        beta_syn = self.model.decoder_syn.weight.cpu().detach().numpy().T
        # beta is of shape (total_topics, vocab_size)
        eval = TopicEval(vocab=self.vocab, text=self.text)
        topics = eval.get_topics(top_n=10, beta=beta_con)
        coherence = eval.topic_coherence(metric='c_v', topics=topics)
        topic_diversity = eval.topic_diversity(topics=topics)
        semantic_purity = eval.semantic_purity(topics=topics)
        logger.info(f"Evaluation completed. Coherence scores for content topics are: {coherence}, topic diversity: {topic_diversity}, semantic purity: {semantic_purity}")
        self.logger['topics_con'] = topics
        self.logger['tq_con'] = {'coherence': coherence,
                                 'topic_diversity': topic_diversity,
                                 'semantic_purity': semantic_purity}
        # dump topics as .txt file
        with open(os.path.join(self.save_path, 'topics_con.txt'), 'w') as f:
            for topic in topics:
                f.write(f"{' '.join(topic)}\n")
            # write evaluation scores
            f.write(f"Coherence scores are: {coherence}, topic diversity: {topic_diversity}, semantic purity: {semantic_purity}")

        topics = eval.get_topics(top_n=10, beta=beta_syn)
        coherence = eval.topic_coherence(metric='c_v', topics=topics)
        topic_diversity = eval.topic_diversity(topics=topics)
        logger.info(f"Evaluation completed. Coherence scores for syntax topics are: {coherence}, topic diversity: {topic_diversity}")
        self.logger['topics_syn'] = topics
        self.logger['tq_syn'] = {'coherence': coherence,
                                 'topic_diversity': topic_diversity}

        pickle.dump(self.logger, open(os.path.join(self.save_path, 'logger.pkl'), 'wb'))
        pickle.dump(self.text, open(os.path.join(self.save_path, 'text.pkl'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.save_path, 'vocab.pkl'), 'wb'))
        # dump topics as .txt file
        with open(os.path.join(self.save_path, 'topics_syn.txt'), 'w') as f:
            for topic in topics:
                f.write(f"{' '.join(topic)}\n")
            # write evaluation scores
            f.write(
                f"Coherence scores are: {coherence}, topic diversity: {topic_diversity}")
        logger.info(f"Results saved at {self.save_path}")

    def train_one_epoch(self):
        all_loss = 0
        all_kl_con = 0
        all_rec_con = 0
        all_kl_syn = 0
        all_rec_syn = 0
        for batch in self.train_dl:
            x = batch['bow'].float().to(self.device)
            self.optim.zero_grad()
            x_recon_con, x_recon_syn, dist_con, dist_syn = self.model(x)
            prior_alpha_con = torch.ones_like(dist_con.concentration).to(self.device) * 0.02
            prior_con = torch.distributions.Dirichlet(prior_alpha_con)
            prior_alpha_syn = torch.ones_like(dist_syn.concentration).to(self.device) * 0.02
            prior_syn = torch.distributions.Dirichlet(prior_alpha_syn)
            kl_con = torch.distributions.kl.kl_divergence(dist_con, prior_con).mean()
            kl_syn = torch.distributions.kl.kl_divergence(dist_syn, prior_syn).mean()
            x_con = x * self.content_vector
            x_syn = x * self.syntax_vector
            recon_loss_con = -torch.sum(x_con * x_recon_con, dim=1).mean()
            recon_loss_syn = -torch.sum(x_syn * x_recon_syn, dim=1).mean()
            kl_loss_curr = 4 * (self.lambda_ * kl_con + (1 - self.lambda_) * kl_syn)
            rec_loss_curr = recon_loss_con + recon_loss_syn
            loss = kl_loss_curr + rec_loss_curr
            all_loss += loss.item()
            all_kl_con += kl_con.item()
            all_rec_con += recon_loss_con.item()
            all_kl_syn += kl_syn.item()
            all_rec_syn += recon_loss_syn.item()
            loss.backward()
            self.optim.step()
        self.logger['train/kld_con'].append(all_kl_con / len(self.train_dl))
        self.logger['train/rec_con'].append(all_rec_con / len(self.train_dl))
        self.logger['train/kld_syn'].append(all_kl_syn / len(self.train_dl))
        self.logger['train/rec_syn'].append(all_rec_syn / len(self.train_dl))
        self.logger['train/loss'].append(all_loss / len(self.train_dl))

    def test_one_epoch(self):
        all_loss = 0
        all_kl_con = 0
        all_rec_con = 0
        all_kl_syn = 0
        all_rec_syn = 0
        with torch.no_grad():
            for batch in self.val_dl:
                x = batch['bow'].float().to(self.device)
                x_recon_con, x_recon_syn, dist_con, dist_syn = self.model(x)
                prior_alpha_con = torch.ones_like(dist_con.concentration).to(self.device) * 0.02
                prior_con = torch.distributions.Dirichlet(prior_alpha_con)
                prior_alpha_syn = torch.ones_like(dist_syn.concentration).to(self.device) * 0.02
                prior_syn = torch.distributions.Dirichlet(prior_alpha_syn)
                kl_con = torch.distributions.kl.kl_divergence(dist_con, prior_con).mean()
                kl_syn = torch.distributions.kl.kl_divergence(dist_syn, prior_syn).mean()
                x_con = x * self.content_vector
                x_syn = x * self.syntax_vector
                recon_loss_con = -torch.sum(x_con * x_recon_con, dim=1).mean()
                recon_loss_syn = -torch.sum(x_syn * x_recon_syn, dim=1).mean()
                kl_loss_curr = 4 * (self.lambda_ * kl_con + (1 - self.lambda_) * kl_syn)
                rec_loss_curr = recon_loss_con + recon_loss_syn
                loss = kl_loss_curr + rec_loss_curr
                all_loss += loss.item()
                all_kl_con += kl_con.item()
                all_rec_con += recon_loss_con.item()
                all_kl_syn += kl_syn.item()
                all_rec_syn += recon_loss_syn.item()
        self.logger['val/kld_con'].append(all_kl_con / len(self.val_dl))
        self.logger['val/rec_con'].append(all_rec_con / len(self.val_dl))
        self.logger['val/kld_syn'].append(all_kl_syn / len(self.val_dl))
        self.logger['val/rec_syn'].append(all_rec_syn / len(self.val_dl))
        self.logger['val/loss'].append(all_loss / len(self.val_dl))
        self.current_val_loss = all_loss / len(self.val_dl)

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'tm.pt'))

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'tm.pt')))
