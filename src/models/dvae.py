# A demo pytorch model with a custom backward pass
import torch
import torch.nn as nn
from torch.distributions import Dirichlet


class DVAE(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_size: int):
        super(DVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=vocab_size, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=500, out_features=num_topics),
            nn.BatchNorm1d(num_features=num_topics, affine=False),
            nn.Softplus()
        )

        self.decoder = nn.Linear(in_features=num_topics, out_features=vocab_size)
        self.decoder_norm = nn.Sequential(
            nn.BatchNorm1d(vocab_size, affine=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        alpha = self.encoder(x)
        alpha = torch.max(torch.tensor(0.00001, device=x.device), alpha)
        dist = Dirichlet(alpha)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        x_recon = self.decoder_norm(self.decoder(z))
        return x_recon, dist


if __name__ == '__main__':
    net = DVAE(num_topics=20, vocab_size=100)
    t = torch.rand(size=(32, 100))
    print(net(t)[0].shape)
