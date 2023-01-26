import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogNormal


class ETM(nn.Module):
    def __init__(self, num_topics, vocab_embeddings):
        super().__init__()
        vocab_size = vocab_embeddings.shape[0]
        self.num_topics = num_topics
        embed_dim = vocab_embeddings.shape[1]

        # encoder, based on: https://jmlr.org/papers/volume20/18-569/18-569.pdf
        self.encoder = nn.Sequential(
            nn.Linear(in_features=vocab_size, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=500, out_features=2 * num_topics),
            nn.BatchNorm1d(num_features=2 * num_topics, affine=False),
            nn.Softplus()
        )

        # embedding decoder
        self.topic_embeddings = nn.Linear(num_topics, embed_dim, bias=False)
        self.word_embeddings = nn.Linear(embed_dim, vocab_size, bias=False)
        # initialize linear layer with pre-trained embeddings
        self.word_embeddings.weight.data.copy_(vocab_embeddings)
        # self.word_embeddings.weight.requires_grad = False
        self.decoder_norm = nn.Sequential(
            nn.BatchNorm1d(vocab_size, affine=False),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # encoder
        alpha = self.encoder(x)
        mu = alpha[:, :self.num_topics]
        sigma = alpha[:, self.num_topics:]
        sigma = torch.max(torch.tensor(0.00001, device=x.device), sigma)
        dist = LogNormal(mu, sigma)
        if self.training:
            dist_sample = dist.rsample()
        else:
            dist_sample = dist.mean
        # decoders
        dist_sample = F.softmax(dist_sample, dim=-1)
        topic_embeddings = self.topic_embeddings(dist_sample)  # (batch_size, 300)
        word_embeddings = self.word_embeddings.weight  # (vocab_size, 300)
        # dot product
        recon = torch.matmul(topic_embeddings, word_embeddings.T)  # (batch_size, vocab_size)
        recon = self.decoder_norm(recon)  # (batch_size, vocab_size)
        return recon, dist


if __name__ == '__main__':
    net = ETM(num_topics=20, vocab_embeddings=torch.rand(100, 50))
    t = torch.rand(size=(32, 100))
    print(net(t)[0].shape)
