from nflows.distributions import Distribution, StandardNormal
from nflows.flows.base import Flow

from . import DensityEstimator
from ..utils import batch_or_dataloader


class NormalizingFlow(DensityEstimator):

    model_type = "nf"

    def __init__(self, dim, transform, base_distribution: Distribution=None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

        if base_distribution is None:
            self.base_distribution = StandardNormal([dim])
        else:
            self.base_distribution = base_distribution

        self._nflow = Flow(
            transform=self.transform,
            distribution=self.base_distribution
        )

    def sample(self, n_samples):
        samples = self._nflow.sample(n_samples)
        return self._inverse_data_transform(samples)

    @batch_or_dataloader()
    def log_prob(self, x):
        # NOTE: Careful with log probability when using _data_transform()
        x = self._data_transform(x)
        log_prob = self._nflow.log_prob(x)

        if len(log_prob.shape) == 1:
            log_prob = log_prob.unsqueeze(1)

        return log_prob
