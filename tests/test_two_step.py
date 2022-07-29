# NOTE: This is not an exhaustive test of all possible two-step models.
#       It is mainly to test the general functionality for a few choices.

import pytest

from two_step_zoo import TwoStepDensityEstimator
from two_step_zoo.density_estimator import EnergyBasedModel
from two_step_zoo.networks import MLP
from .test_gae import autoencoder, bigan
from .test_trainer import small_vae


@pytest.fixture
def small_ebm():
    energy_func = MLP(
        input_dim=pytest.latent_dim,
        hidden_dims=pytest.hidden_dims,
        output_dim=1,
        activation=pytest.activation
    )
    return EnergyBasedModel(energy_func, x_shape=(pytest.latent_dim,))


@pytest.fixture
def bigan_vae(bigan, small_vae):
    return TwoStepDensityEstimator(bigan, small_vae)


@pytest.fixture
def ae_vae(autoencoder, small_vae):
    return TwoStepDensityEstimator(autoencoder, small_vae)


@pytest.fixture
def bigan_ebm(bigan, small_ebm):
    return TwoStepDensityEstimator(bigan, small_ebm)


@pytest.fixture
def ae_ebm(autoencoder, small_ebm):
    return TwoStepDensityEstimator(autoencoder, small_ebm)


two_step_DEs = {"bigan_vae", "ae_vae", "bigan_ebm", "ae_ebm"}


@pytest.fixture
def two_step_de(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("two_step_de", two_step_DEs, indirect=True)
def test_two_step_sample(two_step_de):
    two_step_de.sample(4)


@pytest.mark.parametrize("two_step_de", two_step_DEs, indirect=True)
def test_two_step_log_prob_shape(two_step_de, dataloader):
    log_prob = two_step_de.log_prob(dataloader)
    assert log_prob.shape == (pytest.data_len, 1)


@pytest.mark.parametrize("two_step_de", two_step_DEs, indirect=True)
def test_low_dim_log_prob_shape(two_step_de, dataloader):
    log_prob = two_step_de.low_dim_log_prob(dataloader)
    assert log_prob.shape == (pytest.data_len, 1)


@pytest.mark.parametrize("two_step_de", two_step_DEs, indirect=True)
def test_two_step_rec_error(two_step_de, dataloader):
    two_step_de.rec_error(dataloader)
