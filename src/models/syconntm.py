# A demo pytorch model with a custom backward pass
import torch
import torch.nn as nn
from torch.distributions import Dirichlet


class SyConNTM(nn.Module):
    def __init__(self,
                 num_con_topics: int,
                 num_syn_topics: int,
                 vocab_size: int):
        super(SyConNTM, self).__init__()
        self.num_con_topics = num_con_topics
        self.num_syn_topics = num_syn_topics
        self.encoder = nn.Sequential(
            nn.Linear(in_features=vocab_size, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=500, out_features=num_con_topics + num_syn_topics),
            nn.BatchNorm1d(num_features=num_con_topics + num_syn_topics, affine=False),
            nn.Softplus()
        )

        self.decoder_con = nn.Linear(in_features=num_con_topics, out_features=vocab_size)
        self.decoder_norm_con = nn.Sequential(
            nn.BatchNorm1d(vocab_size, affine=False),
            nn.LogSoftmax(dim=1),
        )

        self.decoder_syn = nn.Linear(in_features=num_syn_topics, out_features=vocab_size)
        self.decoder_norm_syn = nn.Sequential(
            nn.BatchNorm1d(vocab_size, affine=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        alpha = self.encoder(x)
        alpha = torch.max(torch.tensor(0.00001, device=x.device), alpha)
        alpha_con = alpha[:, :self.num_con_topics]
        alpha_syn = alpha[:, self.num_con_topics:]
        dist_con = Dirichlet(alpha_con)
        dist_syn = Dirichlet(alpha_syn)
        if self.training:
            z_con = dist_con.rsample()
            z_syn = dist_syn.rsample()
        else:
            z_con = dist_con.mean
            z_syn = dist_syn.mean
        x_recon_con = self.decoder_norm_con(self.decoder_con(z_con))
        x_recon_syn = self.decoder_norm_syn(self.decoder_syn(z_syn))
        return x_recon_con, x_recon_syn, dist_con, dist_syn


if __name__ == '__main__':
    net = SyConNTM(num_con_topics=10, num_syn_topics=10, vocab_size=100)
    t = torch.rand(size=(32, 100))
    print(net(t)[0].shape)
