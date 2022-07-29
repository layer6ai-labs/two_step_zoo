import pytest
import numpy as np
from nflows.transforms.base import CompositeTransform, MultiscaleCompositeTransform
from nflows.transforms.reshape import SqueezeTransform

from two_step_zoo import GaussianVAE, AdversarialVariationalBayes
from two_step_zoo.networks import MLP, GaussianMixtureLSTM, SimpleFlowTransform
from two_step_zoo.density_estimator import (
    NormalizingFlow, EnergyBasedModel, GaussianMixtureLSTMModel
)


@pytest.fixture
def vae(multihead_encoder, multihead_decoder):
    return GaussianVAE(pytest.latent_dim, multihead_encoder, multihead_decoder)


@pytest.fixture
def avb(multihead_decoder, basic_discriminator):
    encoder = MLP(
        input_dim=pytest.data_dim+pytest.noise_dim,
        hidden_dims=pytest.hidden_dims,
        output_dim=pytest.latent_dim,
        activation=pytest.activation
    )

    return AdversarialVariationalBayes(
        latent_dim=pytest.latent_dim,
        noise_dim=pytest.noise_dim,
        encoder=encoder,
        decoder=multihead_decoder,
        discriminator=basic_discriminator
    )


@pytest.fixture
def ebm():
    energy_func = MLP(
        input_dim=pytest.data_dim,
        hidden_dims=pytest.hidden_dims,
        output_dim=1,
        activation=pytest.activation
    )
    return EnergyBasedModel(energy_func, x_shape=(pytest.data_dim,))


@pytest.fixture
def flow():
    transform = SimpleFlowTransform(
        features_for_mask=pytest.data_dim,
        hidden_features=4,
        num_layers=2,
        num_blocks_per_layer=1
    )
    return NormalizingFlow(pytest.data_dim, transform)


@pytest.fixture
def multiscale_flow():
    # NOTE: Longer test here to set up Multiscale Flow Transform with Squeeze
    #       Flow is the following:
    #       1. Squeeze
    #       2. NSF Transform
    #       3. Split
    #       4. NSF Transform
    transform = MultiscaleCompositeTransform(num_transforms=2)

    post_squeeze_dim = pytest.image_data_shape[1]//2
    post_squeeze_channels = 4*pytest.image_data_shape[0]
    post_squeeze_shape = (post_squeeze_channels, post_squeeze_dim, post_squeeze_dim)

    pre_split_nsf_transform = SimpleFlowTransform(
        features_for_mask=post_squeeze_channels,
        hidden_features=4,
        num_layers=2,
        num_blocks_per_layer=1,
        net="cnn"
    )
    pre_split_transform = CompositeTransform(
        transforms=(SqueezeTransform(), pre_split_nsf_transform)
    )

    post_split_shape = transform.add_transform(
        pre_split_transform,
        post_squeeze_shape
    )

    post_split_transform = SimpleFlowTransform(
        features_for_mask=post_split_shape[0],
        hidden_features=4,
        num_layers=2,
        num_blocks_per_layer=1,
        net="cnn"
    )
    transform.add_transform(
        post_split_transform,
        post_split_shape
    )

    return NormalizingFlow(
        dim=np.prod(pytest.image_data_shape),
        transform=transform
    )


@pytest.fixture
def arm():
    ar_network = GaussianMixtureLSTM(
        input_size=1,
        hidden_size=4,
        num_layers=1,
        k_mixture=3,
    )

    return GaussianMixtureLSTMModel(
        ar_network=ar_network,
        image_height=None,
        input_length=pytest.data_dim,
        data_shape=(pytest.data_dim,)
    )


density_estimators = {"vae", "avb", "ebm", "flow", "arm"}
extended_density_estimators = density_estimators.copy()
extended_density_estimators.add("multiscale_flow")


@pytest.fixture
def density_estimator(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("density_estimator", extended_density_estimators, indirect=True)
def test_de_sample(density_estimator):
    density_estimator.sample(4)


@pytest.mark.parametrize("density_estimator", density_estimators, indirect=True)
def test_log_prob_shape(density_estimator, dataloader):
    log_prob = density_estimator.log_prob(dataloader)
    assert log_prob.shape == (pytest.data_len, 1)


@pytest.mark.parametrize("density_estimator", density_estimators, indirect=True)
def test_de_train_batch(density_estimator, batch):
    density_estimator.set_optimizer(pytest.optim_cfg)
    density_estimator.train_batch(batch[0])


def test_log_prob_shape_multiscale(multiscale_flow, imagelike_dataloader):
    log_prob = multiscale_flow.log_prob(imagelike_dataloader)
    assert log_prob.shape == (pytest.data_len, 1)


def test_multiscale_train_batch(multiscale_flow, imagelike_dataloader):
    def get_single_batch():
        for batch in imagelike_dataloader:
            return batch[0]

    multiscale_flow.set_optimizer(pytest.optim_cfg)
    multiscale_flow.train_batch(get_single_batch())
