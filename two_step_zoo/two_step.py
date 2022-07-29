from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn

from .utils import batch_or_dataloader


class TwoStepDensityEstimator(nn.Module):

    def __init__(self, generalized_autoencoder, density_estimator):
        super().__init__()
        self.generalized_autoencoder = generalized_autoencoder
        self.density_estimator = density_estimator
        self.model_type = ("two_step_"
                            f"{generalized_autoencoder.model_type}_"
                            f"{density_estimator.model_type}")

    def sample(self, n_samples):
        return self.generalized_autoencoder.decode(self.density_estimator.sample(n_samples))

    @batch_or_dataloader()
    def low_dim_log_prob(self, x):
        with torch.no_grad():
            encodings = self.generalized_autoencoder.encode(x)
            return self.density_estimator.log_prob(encodings)

    @batch_or_dataloader()
    def log_prob(self, x):
        with torch.no_grad():
            low_dim_log_prob = self.low_dim_log_prob(x)
            log_det_jtj = self.generalized_autoencoder.log_det_jtj(x)
            return low_dim_log_prob - 0.5*log_det_jtj

    def rec_error(self, *args, **kwargs):
        return self.generalized_autoencoder.rec_error(*args, **kwargs)

    @property
    def device(self):
        return self.generalized_autoencoder.device

    def __getattribute__(self, attr):
        """Redirect other attributes to GAE"""
        gae_attributes = ("encode", "encode_transformed", "decode", "decode_to_transformed",
                          "encoder", "decoder", "rec_error", "latent_dim", "data_min", "data_max",
                          "data_shape")

        if attr in gae_attributes:
            return getattr(self.generalized_autoencoder, attr)
        else:
            return super().__getattribute__(attr)


class TwoStepComponent(nn.Module):
    """Superclass for the GeneralizedAutoencoder and DensityEstimator"""
    _OPTIMIZER_MAP = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop,
    }
    _MIN_WHITEN_STDDEV = 1e-6

    def __init__(
            self,
            flatten=False,
            data_shape=None,
            denoising_sigma=None,
            dequantize=False,
            scale_data=False,
            whitening_transform=False,
            logit_transform=False,
            clamp_samples=False,
        ):
        super().__init__()

        assert not (scale_data and whitening_transform), \
            "Cannot use both a scaling and a whitening transform"

        self.flatten = flatten
        self.data_shape = data_shape
        self.denoising_sigma = denoising_sigma
        self.dequantize = dequantize
        self.scale_data = scale_data
        self.whitening_transform = whitening_transform
        self.logit_transform = logit_transform
        self.clamp_samples = clamp_samples

        # NOTE: Need to set buffers to specific amounts or else they will not be loaded by state_dict
        self.register_buffer("data_min", torch.tensor(0.))
        self.register_buffer("data_max", torch.tensor(1.))

        if whitening_transform:
            whiten_dims = self._get_whiten_dims()

            self.register_buffer("whitening_sigma", torch.ones(*whiten_dims))
            self.register_buffer("whitening_mu", torch.zeros(*whiten_dims))

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Required attribute indicating general model type; e.g., 'vae' or 'nf'"""
        pass

    @property
    def device(self):
        # Consider the model's device to be that of its first parameter
        # (there's no great way to define the `device` of a whole model)
        first_param = next(self.parameters(), None)
        if first_param is not None:
            return first_param.device
        else:
            return None

    def loss(self, x, **kwargs):
        raise NotImplementedError(
            "Implement loss function in child class"
        )

    def train_batch(self, x, max_grad_norm=None, **kwargs):
        self.optimizer.zero_grad()

        loss = self.loss(x, **kwargs)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()

        return {
            "loss": loss
        }

    def set_optimizer(self, cfg):
        self.optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            self.parameters(), lr=cfg["lr"]
        )
        self.num_optimizers = 1

        self.lr_scheduler = self._get_lr_scheduler(
            optim=self.optimizer,
            use_scheduler=cfg.get("use_lr_scheduler", False),
            cfg=cfg
        )

    def _get_lr_scheduler(self, optim, use_scheduler, cfg):
        if use_scheduler:
            # NOTE: Only coding cosine LR scheduler right now
            # NOTE: Use LR scheduling every train step, not every epoch
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optim,
                T_max=cfg["max_epochs"]*(cfg["train_dataset_size"]//cfg["train_batch_size"]),
                eta_min=0.
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optim,
                lr_lambda=lambda step: 1.
            )

        return lr_scheduler

    def set_whitening_params(self, mu, sigma):
        self.whitening_mu = torch.reshape(mu, self._get_whiten_dims())
        self.whitening_sigma = torch.reshape(sigma, self._get_whiten_dims())
        self.whitening_sigma[self.whitening_sigma < self._MIN_WHITEN_STDDEV] = self._MIN_WHITEN_STDDEV

    def _get_whiten_dims(self):
        if self.flatten:
            return (1, np.prod(self.data_shape))
        else:
            return (1, *self.data_shape)

    def _data_transform(self, data):
        if self.flatten:
            data = data.flatten(start_dim=1)
        if self.denoising_sigma is not None and self.training:
            data = data + torch.randn_like(data) * self.denoising_sigma
        if self.dequantize:
            data = data + torch.rand_like(data)
        if self.scale_data:
            data = data / (self.abs_data_max + self.dequantize)
        elif self.whitening_transform:
            data = data - self.whitening_mu
            data = data / self.whitening_sigma
        if self.logit_transform:
            data = torch.logit(data)
        return data

    def _inverse_data_transform(self, data):
        if self.logit_transform:
            data = torch.sigmoid(data)
        if self.scale_data:
            data = data * (self.abs_data_max + self.dequantize)
        elif self.whitening_transform:
            data = data * self.whitening_sigma
            data = data + self.whitening_mu
        if self.dequantize:
            data = torch.floor(data)
        if self.clamp_samples:
            data.clamp_(self.data_min, self.data_max)
        if self.flatten:
            data = data.reshape((-1, *self.data_shape))
        return data

    @property
    def abs_data_max(self):
        return max(self.data_min.abs(), self.data_max.abs())
