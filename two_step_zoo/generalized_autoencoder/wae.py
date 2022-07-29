from itertools import chain
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from . import GeneralizedAutoEncoder
from ..distributions import diagonal_gaussian_sample


class WassersteinAutoEncoder(GeneralizedAutoEncoder):
    model_type = "wae"

    def __init__(
        self,
        latent_dim,
        encoder,
        decoder,
        discriminator,
        _lambda=10.0,
        sigma=1.0,
        **kwargs
    ):
        super().__init__(
            latent_dim,
            encoder,
            decoder,
            **kwargs
        )
        self.discriminator = discriminator
        self._lambda = _lambda
        self.sigma = sigma

    def train_batch(self, x, **kwargs):
        # Train discriminator on batch with encoder and decoder fixed
        self.optimizer[0].zero_grad()
        discriminator_loss = self._discr_error_batch(x).mean()
        discriminator_loss.backward()
        self.optimizer[0].step()
        self.lr_scheduler[0].step()

        # Train encoder and decoder on batch with discriminator fixed
        self.optimizer[1].zero_grad()
        rec_loss = self._rec_error_batch(x).mean()
        rec_loss.backward()
        self.optimizer[1].step()
        self.lr_scheduler[1].step()

        return {
            "discriminator_loss": discriminator_loss,
            "reconstruction_loss": rec_loss
        }

    def _discr_error_batch(self, x):
        x = self._data_transform(x)
        z_q = self.encode_transformed(x)

        mu = torch.zeros_like(z_q)
        z_p = diagonal_gaussian_sample(mu, self.sigma)

        d_z_p = self.discriminator(z_p)
        d_z_q = self.discriminator(z_q)

        ones = torch.ones_like(d_z_q)
        zeros = torch.zeros_like(d_z_p)

        # NOTE: Train discriminator to be positive on encodings z_q
        d_z_p_loss = binary_cross_entropy_with_logits(d_z_p, zeros)
        d_z_q_loss = binary_cross_entropy_with_logits(d_z_q, ones)

        return self._lambda * (d_z_p_loss + d_z_q_loss)

    def _rec_error_batch(self, x):
        # Reconstruction loss
        rec_loss, z_q = self.rec_error(x, return_z=True)

        # Discriminator loss
        d_z_q = self.discriminator(z_q)
        zeros = torch.zeros_like(d_z_q)
        d_loss = binary_cross_entropy_with_logits(d_z_q, zeros)

        return rec_loss + self._lambda * d_loss

    def sample(self, n_samples):
        mu = torch.zeros(n_samples, self.latent_dim).to(self.device)
        z_p = diagonal_gaussian_sample(mu, self.sigma)
        x = self.decode(z_p)
        return x

    def set_optimizer(self, cfg):
        disc_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            self.discriminator.parameters(), lr=cfg["disc_lr"]
        )
        rec_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=cfg["rec_lr"]
        )
        self.optimizer = [disc_optimizer, rec_optimizer]
        self.num_optimizers = 2

        disc_lr_scheduler = self._get_lr_scheduler(
            optim=disc_optimizer,
            use_scheduler=cfg.get("use_disc_lr_scheduler", False),
            cfg=cfg
        )
        rec_lr_scheduler = self._get_lr_scheduler(
            optim=rec_optimizer,
            use_scheduler=cfg.get("use_rec_lr_scheduler", False),
            cfg=cfg
        )
        self.lr_scheduler = [disc_lr_scheduler, rec_lr_scheduler]
