import pytest

from two_step_zoo.generalized_autoencoder import AutoEncoder, BiGAN, WassersteinAutoEncoder
from two_step_zoo.networks.neural_networks import MLP
from .test_de import density_estimators, vae, avb


@pytest.fixture
def autoencoder(basic_encoder, basic_decoder):
    return AutoEncoder(pytest.latent_dim, basic_encoder, basic_decoder)


@pytest.fixture
def bigan(basic_encoder, basic_decoder, basic_discriminator):
    return BiGAN(pytest.latent_dim, basic_encoder, basic_decoder, basic_discriminator)


@pytest.fixture
def wae(basic_encoder, basic_decoder):
    discriminator = MLP(
        input_dim=pytest.latent_dim,
        hidden_dims=pytest.hidden_dims,
        output_dim=1,
        activation=pytest.activation
    )
    return WassersteinAutoEncoder(pytest.latent_dim, basic_encoder, basic_decoder, discriminator)


generalized_autoencoders = {"vae", "avb", "autoencoder", "bigan", "wae"}


@pytest.fixture
def generalized_autoencoder(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("generalized_autoencoder", generalized_autoencoders, indirect=True)
def test_gae_rec_error(generalized_autoencoder, dataloader):
    generalized_autoencoder.rec_error(dataloader)


@pytest.mark.parametrize("generalized_autoencoder", generalized_autoencoders - density_estimators,
                         indirect=True)
def test_gae_train_batch(generalized_autoencoder, batch):
    generalized_autoencoder.set_optimizer(pytest.optim_cfg)
    generalized_autoencoder.train_batch(batch[0])
