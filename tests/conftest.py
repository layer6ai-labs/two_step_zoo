"""
Operations to be shared across multiple tests
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from two_step_zoo.datasets import Sphere, SupervisedDataset
from two_step_zoo.networks import MLP


def pytest_configure():
    pytest.latent_dim = 2
    pytest.hidden_dims = [5, 5]
    pytest.batch_size = 5
    pytest.data_dim = 4
    pytest.data_len = 64
    pytest.image_data_shape = [3, 32, 32] # For Multiscale Flow
    pytest.noise_dim = 1 # For AVB
    pytest.activation = nn.ReLU
    pytest.optim_cfg = {
        "optimizer": "sgd",
        "lr": 1e-3,
        "disc_lr": 1e-3, # For AVB
        "nll_lr": 1e-3, # For AVB
        "ge_lr": 1e-3, # For BiGAN
        "rec_lr": 1e-3, # For WAE
    }


@pytest.fixture
def dataloader():
    manifold_dim = 2

    dataset = Sphere("sphere", "train", manifold_dim, pytest.data_dim, pytest.data_len)
    dataloader = DataLoader(dataset, pytest.batch_size)
    return dataloader


@pytest.fixture
def imagelike_dataloader():
    dataset = SupervisedDataset(
        name="Test Image Dataset",
        role="train",
        x=torch.rand((pytest.data_len, *pytest.image_data_shape))
    )
    dataloader = DataLoader(dataset, pytest.batch_size)
    return dataloader


@pytest.fixture
def batch(dataloader):
    for batch in dataloader:
        return batch


@pytest.fixture
def basic_encoder():
    return MLP(
        input_dim=pytest.data_dim,
        hidden_dims=pytest.hidden_dims,
        output_dim=pytest.latent_dim,
        activation=pytest.activation
    )


@pytest.fixture
def basic_decoder():
    return MLP(
        input_dim=pytest.latent_dim,
        hidden_dims=pytest.hidden_dims,
        output_dim=pytest.data_dim,
        activation=pytest.activation
    )


@pytest.fixture
def multihead_encoder():
    return MLP(
        input_dim=pytest.data_dim,
        hidden_dims=pytest.hidden_dims,
        output_dim=2*pytest.latent_dim,
        activation=pytest.activation,
        output_split_sizes=[pytest.latent_dim, pytest.latent_dim]
    )


@pytest.fixture
def multihead_decoder():
    return MLP(
        input_dim=pytest.latent_dim,
        hidden_dims=pytest.hidden_dims,
        output_dim=2*pytest.data_dim,
        activation=pytest.activation,
        output_split_sizes=[pytest.data_dim, pytest.data_dim]
    )


@pytest.fixture
def basic_discriminator():
    return MLP(
        input_dim=pytest.data_dim+pytest.latent_dim,
        hidden_dims=pytest.hidden_dims,
        output_dim=1,
        activation=pytest.activation
    )
