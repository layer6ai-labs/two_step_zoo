import os
import pathlib
import subprocess

import pytest


flags = {
    "one_step": (
        "--dataset mnist "
        "--config max_epochs=2 "
        "--model vae "
        "--config data_shape=(1,28,28) "
        "--config data_dim=784 "
        "--config flatten=True "
        "--config encoder_net=mlp "
        "--config encoder_hidden_dims=[5] "
        "--config decoder_net=mlp "
        "--config decoder_hidden_dims=[5] "
        "--config scale_data=True "
        "--config valid_metrics=['loss'] "
        "--config test_metrics=['loss'] "
    ),

    "two_step": (
        "--dataset mnist "
        "--gae-config data_shape=(1,28,28) "
        "--gae-config data_dim=784 "
        "--gae-config flatten=True "
        "--gae-config scale_data=True "
        "--gae-config max_epochs=1 "
        "--de-config max_epochs=1 "
        "--gae-config valid_metrics=[] "
        "--gae-config test_metrics=[] "
        "--gae-config early_stopping_metric=None "
        "--de-config valid_metrics=[] "
        "--de-config test_matrics=[] "
        "--de-config early_stopping_metric=None "
        "--shared-config test_metrics=['l2_reconstruction_error'] "
        "--shared-config max_epochs=2"
    ),

    "alternate_batch": (
        "--shared-config sequential_training=False"
    ),

    "alternate_epoch": (
        "--shared-config sequential_training=False "
        "--shared-config alternate_by_epoch=True"
    ),

    "reload": "--load-dir",

    "gae_nets": (
        "--gae-config encoder_net=mlp "
        "--gae-config encoder_hidden_dims=[5] "
        "--gae-config decoder_net=mlp "
        "--gae-config decoder_hidden_dims=[5] "
    ),

    "vae_vae": "--gae-model vae --de-model vae",

    "avb_avb": "--gae-model avb --de-model avb",

    "ae_flow": (
        "--gae-model ae --de-model flow --de-config hidden_units=5 "
        "--de-config num_layers=1 --de-config num_blocks_per_layer=1 "
    ),

    "bigan_ebm": "--gae-model bigan --de-model ebm",

    "wae_arm": "--gae-model wae --de-model arm"
}

for key, val in flags.items():
    flags[key] = val.split()


def get_latest_run():
    run_dirs = pathlib.Path("runs").glob("*")
    return max(run_dirs, key=os.path.getctime)


@pytest.mark.cmd
@pytest.mark.parametrize("model_name", ["vae_vae", "avb_avb", "ae_flow", "bigan_ebm", "wae_arm"])
def test_sequential_training(model_name):
    # Test that training runs without error
    train_run = subprocess.run(
        ["python", "main.py"] + flags["two_step"] + flags[model_name] + flags["gae_nets"])
    assert train_run.returncode == 0

    # Test that model can be reloaded without error
    reload_run = subprocess.run(
        ["python", "main.py"] + flags["reload"] + [get_latest_run()])
    assert reload_run.returncode == 0


@pytest.mark.cmd
def test_single_training():
    # Test that single training runs without error
    train_run = subprocess.run(
        ["python", "single_main.py"] + flags["one_step"])
    assert train_run.returncode == 0


@pytest.mark.cmd
@pytest.mark.parametrize("train_type", ["alternate_batch", "alternate_epoch"])
def test_alternating_training(train_type):
    # Test that alternating training runs without error
    train_run = subprocess.run(
        ["python", "main.py"] + flags["two_step"] + flags[train_type] + flags["vae_vae"]
        + flags["gae_nets"])
    assert train_run.returncode == 0
