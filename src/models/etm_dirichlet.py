import torch
import torch.nn as nn
from torch.distributions import Dirichlet


class ETMDirichlet(nn.Module):
    def __init__(self, num_topics, vocab_embeddings):
        super().__init__()
        vocab_size = vocab_embeddings.shape[0]
        embed_dim = vocab_embeddings.shape[1]

        # encoder, based on: https://jmlr.org/papers/volume20/18-569/18-569.pdf
        self.encoder = nn.Sequential(
            nn.Linear(in_features=vocab_size, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=500, out_features=num_topics),
            nn.BatchNorm1d(num_features=num_topics, affine=False),
            nn.Softplus()
        )

        # embedding decoder
        self.topic_embeddings = nn.Linear(num_topics, embed_dim)
        self.word_embeddings = nn.Linear(embed_dim, vocab_size)
        # initialize linear layer with pre-trained embeddings
        self.word_embeddings.weight.data.copy_(vocab_embeddings)
        # self.word_embeddings.weight.requires_grad = False
        self.decoder_norm = nn.Sequential(
            nn.BatchNorm1d(vocab_size, affine=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        # encoder
        alpha = self.encoder(x)
        # split into semantic and syntax
        alpha = torch.max(torch.tensor(0.00001, device=x.device), alpha)
        # sample from dirichlet
        dist = Dirichlet(alpha)
        if self.training:
            dist_sample = dist.rsample()
        else:
            dist_sample = dist.mean
        # decoders
        topic_embeddings = self.topic_embeddings(dist_sample)  # (batch_size, 300)
        word_embeddings = self.word_embeddings.weight  # (vocab_size, 300)
        # dot product
        recon = torch.matmul(topic_embeddings, word_embeddings.T)  # (batch_size, vocab_size)
        recon = self.decoder_norm(recon)  # (batch_size, vocab_size)
        return recon, dist


if __name__ == '__main__':
    net = ETMDirichlet(num_topics=20, vocab_embeddings=torch.rand(100, 300))
    t = torch.rand(size=(32, 100))
    print(net(t)[0].shape)
