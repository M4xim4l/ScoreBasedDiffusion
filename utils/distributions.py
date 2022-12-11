import torch
from torch.distributions import MultivariateNormal

class SumOfGaussians:
    def __init__(self, means, covariances, weights):
        self.means = means
        self.covariances = covariances
        self.weights = torch.FloatTensor(weights)
        self.cum_weights = torch.cat([torch.zeros(1),torch.cumsum(self.weights, dim=0)])


        self.multivar_norms = []
        for mean, covar in zip(means, covariances):
            self.multivar_norms.append(MultivariateNormal(mean, covar)
)
    def log_prob(self, x):
        return torch.log(self.prob(x))

    def prob(self, x):
        p_x = sum([w * distribution.log_prob(x).exp() for w, distribution in zip(self.weights, self.multivar_norms)])
        return p_x

    def sample(self, N):
        rand_vals = torch.rand(N)[:,None]
        rand_vals_minus_cum_weights = rand_vals - self.cum_weights[None,:]
        larger_than_prev = (rand_vals_minus_cum_weights > 0)[:, :-1]
        smaller_than_next = (rand_vals_minus_cum_weights <= 0)[:, 1:]
        distribution_idcs = larger_than_prev & smaller_than_next

        samples = torch.zeros((N, len(self.means[0])))
        for dist_idx, dist in enumerate(self.multivar_norms):
            samples += distribution_idcs[:, dist_idx][:, None] * dist.sample((N,))
        return samples

