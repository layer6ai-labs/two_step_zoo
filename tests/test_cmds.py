import subprocess

import pytest

from .cmd_helpers import model_flags, flags, get_latest_run


@pytest.mark.cmd_exhaustive
@pytest.mark.parametrize("train_type", ["sequential", "alternate_batch", "alternate_epoch"])
@pytest.mark.parametrize("gae", ["vae", "avb", "ae", "bigan", "wae"])
@pytest.mark.parametrize("de", ["vae", "avb", "flow", "ebm", "arm"])
def test_all_two_step(train_type, gae, de):
    # Test that training runs without error
    run_flags = (
        ["python", "main.py"] + flags["two_step"] + model_flags(gae, "gae") + model_flags(de, "de")
        + flags.get(train_type, [])
    )
    train_run = subprocess.run(run_flags)
    assert train_run.returncode == 0

    # Test that model can be reloaded and trained without error
    reload_run = subprocess.run(run_flags + flags["reload_two_step"] + [get_latest_run()])
    assert reload_run.returncode == 0


@pytest.mark.cmd_exhaustive
@pytest.mark.parametrize("gae", ["vae", "avb", "ae", "bigan", "wae"])
@pytest.mark.parametrize("de", ["vae", "avb", "flow", "ebm", "arm"])
def test_all_pretrained_gae(gae, de):
    # Test that training runs without error
    train_run = subprocess.run(
        ["python", "single_main.py"] + flags["one_step"] + model_flags(gae, train_type="one_step")
        + ["--is-gae"] # TODO: remove line when flag is deprecated
    )
    assert train_run.returncode == 0

    # Test that GAE can be reloaded and frozen while a DE is trained without error
    run_flags = (
        ["python", "main.py"] + flags["two_step"] + model_flags(gae, "gae") + model_flags(de, "de")
          + flags["load_gae"] + flags["reload_two_step"] + [get_latest_run()]
    )
    reload_run = subprocess.run(run_flags + flags["freeze"])
    assert reload_run.returncode == 0

    # Test that same GAE can be reloaded and trained with a DE without error
    reload_run = subprocess.run(run_flags)
    assert reload_run.returncode == 0


@pytest.mark.cmd
@pytest.mark.cmd_exhaustive
@pytest.mark.parametrize("de", ["vae", "avb", "flow", "ebm", "arm"])
def test_one_step_de(de):
    run_flags = (
        ["python", "single_main.py"] + flags["one_step"] + model_flags(de, train_type="one_step")
    )
    # Test that training runs without error
    train_run = subprocess.run(run_flags)
    assert train_run.returncode == 0

    # Test that model can be reloaded and trained without error
    reload_run = subprocess.run(run_flags + flags["reload_one_step"] + [get_latest_run()])
    assert reload_run.returncode == 0


@pytest.mark.cmd
@pytest.mark.parametrize("train_type", ["sequential", "alternate_batch", "alternate_epoch"])
def test_two_step(train_type):
    test_all_two_step(train_type, "vae", "vae")


@pytest.mark.cmd
@pytest.mark.parametrize("gae", ["vae", "avb", "ae", "bigan", "wae"])
def test_pretrained_gae(gae):
    test_all_pretrained_gae(gae, "vae")
