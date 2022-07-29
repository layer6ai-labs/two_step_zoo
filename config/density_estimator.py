def get_base_config(dataset, standalone):
    if standalone:
        standalone_info = {
            "train_batch_size": 128,
            "valid_batch_size": 128,
            "test_batch_size": 128,

            "make_valid_loader": True,

            "data_root": "data/",
            "logdir_root": "runs/"
        }
        scale_data = True
        whitening_transform = False
    else:
        standalone_info = {}
        scale_data = False
        whitening_transform = True

    return {
        "flatten": True,
        "denoising_sigma": None,
        "dequantize": False,
        "scale_data": scale_data,
        "whitening_transform": whitening_transform,

        "optimizer": "adam",
        "lr": 0.001,
        "use_lr_scheduler": False,
        "max_epochs": 100,
        "max_grad_norm": 10,

        "early_stopping_metric": None,
        "max_bad_valid_epochs": None,

        # NOTE: A validation metric should indicate better performance as it decreases.
        #       Thus, log_likelihood is not an appropriate validation metric.
        "valid_metrics": ["loss"],
        "test_metrics": ["log_likelihood"],

        **standalone_info
    }


def get_arm_config(dataset, standalone):
    arm_base = {
        "k_mixture": 10,

        "flatten": False,
        "early_stopping_metric": "loss",
        "max_bad_valid_epochs": 10,
        "use_lr_scheduler": True
    }

    if standalone:
        hidden_size = 256
        num_layers = 2
    else:
        hidden_size = 128
        num_layers = 1

    net_config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }

    return{
        **arm_base,
        **net_config
    }


def get_avb_config(dataset, standalone):
    return {
        "max_epochs": 100,

        "noise_dim": 128,
        "latent_dim": 20,
        "encoder_net": "mlp",
        "decoder_net": "mlp",
        "encoder_hidden_dims": [256],
        "decoder_hidden_dims": [256],
        "discriminator_hidden_dims": [256, 256],

        "single_sigma": True,

        "input_sigma": 3.,
        "prior_sigma": 1.,

        "lr": None,
        "disc_lr": 0.001,
        "nll_lr": 0.001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_nll_lr_scheduler": True,
    }


def get_ebm_config(dataset, standalone):
    if standalone:
        net = "mlp" if dataset in ["mnist", "fashion-mnist"] else "cnn"
        lr = 0.0003
        x_lims = (0, 1)
        loss_alpha = 1.0
        spectral_norm = True
        if net == "mlp":
            energy_func_hidden_dims = [256, 128]
    else:
        net = "mlp"
        x_lims = (-1, 1)
        energy_func_hidden_dims = [64, 32]
        lr = 0.001
        loss_alpha = 0.1
        spectral_norm = False

    ebm_base = {
        "max_length_buffer": 8192,
        "x_lims": x_lims,
        "ld_steps": 60,
        "ld_step_size": 10,
        "ld_eps_new": 0.05,
        "ld_sigma": 0.005,
        "ld_grad_clamp": 0.03,
        "loss_alpha": loss_alpha,

        "scale_data": True,
        "whitening_transform": False,
        "spectral_norm": spectral_norm,

        "lr": lr,
        "max_grad_norm": 1.0,
    }

    if net == "mlp":
        net_config = {
            "net": "mlp",
            "energy_func_hidden_dims": energy_func_hidden_dims
        }

    elif net == "cnn":
        net_config = {
            "net": "cnn",
            "energy_func_hidden_channels": [64, 64, 32, 32],
            "energy_func_kernel_size": [3, 3, 3, 3],
            "energy_func_stride": [1, 1, 1, 1],

            "flatten": False
        }

    return {
        **ebm_base,
        **net_config
    }


def get_flow_config(dataset, standalone):
    if standalone:
        hidden_units = 128
        lr = 0.0005
        standalone_info = {
            "early_stopping_metric": "loss",
            "max_bad_valid_epochs": 30,
        }
    else:
        hidden_units = 64
        lr = 0.001
        standalone_info = {}
    flow_config = {
        "scale_data": False,
        "whitening_transform": True,

        "transform": "simple_nsf",
        "hidden_units": hidden_units,
        "num_layers": 4,
        "num_blocks_per_layer": 3,

        "lr": lr,
    }
    return {
        **flow_config,
        **standalone_info,
    }


def get_vae_config(dataset, standalone):
    vae_base = {
        "latent_dim": 20,
        "use_lr_scheduler": False,

        "single_sigma": True,
    }

    net_config = {
        "encoder_net": "mlp",
        "encoder_hidden_dims": [256],

        "decoder_net": "mlp",
        "decoder_hidden_dims": [256],

        "flatten": True
    }

    return {
        **vae_base,
        **net_config,
    }


DE_CFG_MAP = {
    "base": get_base_config,
    "arm": get_arm_config,
    "avb": get_avb_config,
    "ebm": get_ebm_config,
    "flow": get_flow_config,
    "vae": get_vae_config
}
