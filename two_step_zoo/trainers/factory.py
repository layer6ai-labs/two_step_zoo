from .single_trainer import SingleTrainer
from .two_step_trainer import SequentialTrainer, AlternatingEpochTrainer, AlternatingIterationTrainer


def get_single_trainer(
        module,
        ckpt_prefix,
        writer,
        cfg,
        train_loader,
        valid_loader,
        test_loader,
        evaluator,
        only_test=False,
):
    return SingleTrainer(
        module=module,
        ckpt_prefix=ckpt_prefix,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        writer=writer,
        max_epochs=cfg["max_epochs"],
        early_stopping_metric=cfg["early_stopping_metric"],
        max_bad_valid_epochs=cfg["max_bad_valid_epochs"],
        max_grad_norm=cfg["max_grad_norm"],
        evaluator=evaluator,
        only_test=only_test
    )


def get_trainer(
        two_step_module,

        writer,

        gae_cfg,
        de_cfg,
        shared_cfg,

        train_loader,
        valid_loader,
        test_loader,

        gae_evaluator,
        de_evaluator,
        shared_evaluator,

        load_best_valid_first,

        pretrained_gae_path,
        freeze_pretrained_gae,

        only_test=False
):
    gae_trainer = get_single_trainer(
        module=two_step_module.generalized_autoencoder,
        ckpt_prefix="gae",
        writer=writer,
        cfg=gae_cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        evaluator=gae_evaluator,
        only_test=only_test,
    )
    de_trainer = get_single_trainer(
        module=two_step_module.density_estimator,
        ckpt_prefix="de",
        writer=writer,
        cfg=de_cfg,
        train_loader=None,
        valid_loader=None,
        test_loader=None,
        evaluator=de_evaluator,
        only_test=only_test,
    )

    checkpoint_load_list = ["best_valid", "latest"] if load_best_valid_first else ["latest", "best_valid"]

    if shared_cfg["sequential_training"]:
        return SequentialTrainer(
            gae_trainer=gae_trainer,
            de_trainer=de_trainer,
            writer=writer,
            evaluator=shared_evaluator,
            checkpoint_load_list=checkpoint_load_list,
            pretrained_gae_path=pretrained_gae_path,
            freeze_pretrained_gae=freeze_pretrained_gae,
            only_test=only_test
        )

    elif shared_cfg["alternate_by_epoch"]:
        trainer_class = AlternatingEpochTrainer
    else:
        trainer_class = AlternatingIterationTrainer

    return trainer_class(
        two_step_module=two_step_module,
        gae_trainer=gae_trainer,
        de_trainer=de_trainer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        writer=writer,
        max_epochs=shared_cfg["max_epochs"],
        early_stopping_metric=shared_cfg["early_stopping_metric"],
        max_bad_valid_epochs=shared_cfg["max_bad_valid_epochs"],
        max_grad_norm=shared_cfg["max_grad_norm"],
        evaluator=shared_evaluator,
        checkpoint_load_list=checkpoint_load_list,
        pretrained_gae_path=pretrained_gae_path,
        freeze_pretrained_gae=freeze_pretrained_gae,
        only_test=only_test
    )
