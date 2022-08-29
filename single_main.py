#!/usr/bin/env python3

import argparse
import pprint
import torch

from config import get_single_config, load_config, parse_config_arg
from two_step_zoo import (
    get_single_module, get_single_trainer, get_loaders_from_config,
    get_writer, get_evaluator, get_ood_evaluator
)


parser = argparse.ArgumentParser(
    description="Single Density Estimation or Generalized Autoencoder Training Module"
)

parser.add_argument("--dataset", type=str,
    help="Dataset to train on. Required if load_dir not specified.")
parser.add_argument("--model", type=str,
    help="Model to train. Required if load_dir not specified.")
parser.add_argument("--is-gae", action="store_true",
    help="Indicates that we are training a generalized autoencoder.")

parser.add_argument("--load-dir", type=str, default="",
    help="Directory to load from.")
parser.add_argument("--max-epochs-loaded", type=int,
    help="New maximum shared epochs for loaded model.")
parser.add_argument("--load-best-valid-first", action="store_true",
    help="Load the best_valid checkpoint first")

parser.add_argument("--config", default=[], action="append",
    help="Override config entries. Specify as `key=value`.")

parser.add_argument("--only-test", action="store_true",
    help="Only perform a test, no training.")

parser.add_argument("--test-ood", action="store_true",
    help="Perform an OOD test.")

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"


if args.load_dir:
    # NOTE: Not updating config values using cmd line arguments (besides max_epochs)
    #       when loading a run.
    cfg = load_config(
        args=args
    )
else:
    cfg = get_single_config(
        dataset=args.dataset,
        model=args.model,
        gae=args.is_gae,
        standalone=True
    )
    cfg = {**cfg, **dict(parse_config_arg(kv) for kv in args.config)}


pprint.sorted = lambda x, key=None: x
pp = pprint.PrettyPrinter(indent=4)
print(10*"-" + "cfg" + 10*"-")
pp.pprint(cfg)


train_loader, valid_loader, test_loader = get_loaders_from_config(cfg)
writer = get_writer(args, cfg=cfg)


module = get_single_module(
    cfg,
    data_dim=cfg["data_dim"],
    data_shape=cfg["data_shape"],
    train_dataset_size=cfg["train_dataset_size"]
).to(device)


if args.test_ood or "likelihood_ood_acc" in cfg["test_metrics"]:
    evaluator = get_ood_evaluator(
        module,
        cfg=cfg,
        include_low_dim=False,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_loader=train_loader,
        savedir=writer.logdir
    )
else:
    if cfg["early_stopping_metric"] == "fid" and "fid" not in cfg["valid_metrics"]:
        cfg["valid_metrics"].append("fid")

    evaluator = get_evaluator(
        module,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=cfg["valid_metrics"],
        test_metrics=cfg["test_metrics"],
        **cfg.get("metric_kwargs", {}),
    )


trainer = get_single_trainer(
    module=module,
    ckpt_prefix="gae" if cfg["gae"] else "de",
    writer=writer,
    cfg=cfg,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    evaluator=evaluator,
    only_test=args.only_test
)

checkpoint_load_list = ["latest", "best_valid"]
if args.load_best_valid_first: checkpoint_load_list = checkpoint_load_list[::-1]
for ckpt in checkpoint_load_list:
    try:
        trainer.load_checkpoint(ckpt)
        break
    except FileNotFoundError:
        print(f"Did not find {ckpt} checkpoint")

trainer.train()
