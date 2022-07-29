import pytest
import tempfile

from two_step_zoo import Writer, GaussianVAE, TwoStepDensityEstimator
from two_step_zoo.evaluators import Evaluator
from two_step_zoo.networks import MLP
from two_step_zoo.trainers.single_trainer import SingleTrainer
from two_step_zoo.trainers.two_step_trainer import (
    SequentialTrainer, AlternatingEpochTrainer, AlternatingIterationTrainer
)
from .test_de import vae


def get_single_trainer(module, writer, dataloader, gae):
    return SingleTrainer(
        module,
        ckpt_prefix="gae" if gae else "de",
        train_loader=dataloader,
        valid_loader=dataloader,
        test_loader=dataloader,
        writer=writer,
        max_epochs=2
    )


def get_alternating_trainer(vae, small_vae, writer, dataloader, iteration=True):
    trainer_class = AlternatingIterationTrainer if iteration else AlternatingEpochTrainer
    two_step_module = TwoStepDensityEstimator(vae, small_vae)

    evaluator = Evaluator(
        module=two_step_module,
        valid_loader=dataloader,
        test_loader=dataloader
    )

    return trainer_class(
        two_step_module=two_step_module,
        gae_trainer=get_single_trainer(vae, writer, dataloader, gae=True),
        de_trainer=get_single_trainer(small_vae, writer, dataloader, gae=False),
        train_loader=dataloader,
        valid_loader=dataloader,
        test_loader=dataloader,
        writer=writer,
        max_epochs=2,
        early_stopping_metric=None,
        max_bad_valid_epochs=None,
        max_grad_norm=None,
        evaluator=evaluator,
        checkpoint_load_list=[]
    )


def get_small_multihead_mlp(dim, hidden_dims, activation):
    return MLP(
        input_dim=dim,
        hidden_dims=hidden_dims,
        output_dim=2*dim,
        activation=activation,
        output_split_sizes=[dim, dim]
    )


@pytest.fixture
def writer():
    temp_dir = tempfile.TemporaryDirectory()
    return Writer(logdir=temp_dir.name, make_subdir=False, tag_group="")


@pytest.fixture
def small_vae():
    encoder = get_small_multihead_mlp(pytest.latent_dim, pytest.hidden_dims, pytest.activation)
    decoder = get_small_multihead_mlp(pytest.latent_dim, pytest.hidden_dims, pytest.activation)
    return GaussianVAE(pytest.latent_dim, encoder, decoder)


@pytest.fixture
def single_trainer(vae, writer, dataloader):
    vae.set_optimizer(pytest.optim_cfg)

    return get_single_trainer(vae, writer, dataloader, gae=True)


@pytest.fixture
def sequential_trainer(vae, small_vae, writer, dataloader):
    vae.set_optimizer(pytest.optim_cfg)
    small_vae.set_optimizer(pytest.optim_cfg)

    return SequentialTrainer(
        gae_trainer=get_single_trainer(vae, writer, dataloader, gae=True),
        de_trainer=get_single_trainer(small_vae, writer, dataloader, gae=False),
        writer=writer,
        evaluator=None,
        checkpoint_load_list=[]
    )


@pytest.fixture
def alternating_iteration_trainer(vae, small_vae, writer, dataloader):
    vae.set_optimizer(pytest.optim_cfg)
    small_vae.set_optimizer(pytest.optim_cfg)

    return get_alternating_trainer(vae, small_vae, writer, dataloader)


@pytest.fixture
def alternating_epoch_trainer(vae, small_vae, writer, dataloader):
    vae.set_optimizer(pytest.optim_cfg)
    small_vae.set_optimizer(pytest.optim_cfg)

    return get_alternating_trainer(vae, small_vae, writer, dataloader, iteration=False)


non_sequential_trainers = {
    # All trainer besides SequentialTrainer as that has fundamentally different behaviour
    "single_trainer",
    "alternating_iteration_trainer",
    "alternating_epoch_trainer"
}


@pytest.fixture
def non_sequential_trainer(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("non_sequential_trainer", non_sequential_trainers, indirect=True)
def test_train_for_epoch(non_sequential_trainer):
    non_sequential_trainer.train_for_epoch()


@pytest.mark.parametrize("non_sequential_trainer", non_sequential_trainers, indirect=True)
def test_checkpointing(non_sequential_trainer):
    non_sequential_trainer.train_for_epoch()
    non_sequential_trainer.write_checkpoint("test")
    non_sequential_trainer.load_checkpoint("test")


def sequential_train_for_epoch(sequential_trainer):
    sequential_trainer.gae_trainer.train_for_epoch()
    sequential_trainer._set_de_loaders()
    sequential_trainer.de_trainer.train_for_epoch()


def test_sequential_train_for_epoch(sequential_trainer):
    sequential_train_for_epoch(sequential_trainer)


def test_sequential_checkpointing(sequential_trainer):
    sequential_train_for_epoch(sequential_trainer)
    sequential_trainer.write_checkpoint("test")
    sequential_trainer.load_checkpoint("test")
