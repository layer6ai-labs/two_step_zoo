"""
Helpers for constructing command line flags and other arguments in command line tests.
"""
import os
import pathlib
import tempfile


run_dir = pathlib.Path(tempfile.TemporaryDirectory().name)


flags = {
    "one_step": (
        "--dataset mnist "
        "--config max_epochs=2 "
        "--config data_shape=(1,28,28) "
        "--config data_dim=784 "
        "--config scale_data=True "
        "--config whitening_transform=False "
        "--config valid_metrics=['loss'] "
        "--config test_metrics=['loss'] "
        "--config early_stopping_metric=None "
        f"--config logdir_root='{str(run_dir)}' "
    ),

    "two_step": (
        "--dataset mnist "
        "--gae-config data_shape=(1,28,28) "
        "--gae-config data_dim=784 "
        "--gae-config flatten=True "
        "--gae-config scale_data=True "
        "--gae-config max_epochs=2 "
        "--de-config max_epochs=2 "
        "--gae-config valid_metrics=[] "
        "--gae-config test_metrics=[] "
        "--gae-config early_stopping_metric=None "
        "--de-config valid_metrics=[] "
        "--de-config test_matrics=[] "
        "--de-config early_stopping_metric=None "
        "--shared-config early_stopping_metric=None "
        "--shared-config test_metrics=['l2_reconstruction_error'] "
        "--shared-config max_epochs=2 "
        f"--shared-config logdir_root='{str(run_dir)}' "
    ),

    "alternate_batch": (
        "--shared-config sequential_training=False"
    ),

    "alternate_epoch": (
        "--shared-config sequential_training=False "
        "--shared-config alternate_by_epoch=True"
    ),

    "reload_one_step": "--max-epochs-loaded=4 --load-dir",

    "reload_two_step": "--max-epochs-loaded-gae=4 --max-epochs-loaded-de=4 --load-dir",

    "load_gae": "--load-pretrained-gae ",

    "freeze": "--freeze-pretrained-gae",
}


for key, val in flags.items():
    flags[key] = val.split()


model_configs = {
    "vae": ["flatten=True"],
    "avb": ["flatten=True"],
    "ebm": ["flatten=True"],
    "flow": ["hidden_units=5", "num_layers=1", "num_blocks_per_layer=1", "whitening=False",
             "flatten=True"],
    "arm": ["flatten=False"],
    "wae": ["flatten=True"],
}


def model_flags(model_type, step=None, train_type="two_step"):
    assert (train_type == "one_step" and step == None
            or train_type == "two_step" and step in ("gae", "de"))

    flags = []

    if train_type == "one_step":
        flags.append("--model")
        config_flag = "--config"
    elif step == "gae":
        flags.append("--gae-model")
        config_flag = "--gae-config"
    elif step == "de":
        flags.append("--de-model")
        config_flag = "--de-config"

    flags.append(model_type)

    for config in model_configs.get(model_type, []):
        flags += [config_flag, config]

    if train_type == "one_step" or step == "gae":
        for config in ["encoder_net=mlp", "encoder_hidden_dims=[5]", "decoder_net=mlp",
                       "decoder_hidden_dims=[5]"]:
            flags += [config_flag, config]

    return flags


def get_latest_run():
    run_dirs = run_dir.glob("*")
    return max(run_dirs, key=os.path.getctime)
