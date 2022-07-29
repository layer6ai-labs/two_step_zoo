def get_shared_config(dataset):
    return {
        "dataset": dataset,

        "sequential_training": True,
        "alternate_by_epoch": False,

        "max_epochs": 100,
        "early_stopping_metric": None,
        "max_bad_valid_epochs": None,
        "max_grad_norm": None,

        "make_valid_loader": False,

        "data_root": "data/",
        "logdir_root": "runs/",

        "train_batch_size": 128,
        "valid_batch_size": 128,
        "test_batch_size": 128,

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error", "fid"],
    }
