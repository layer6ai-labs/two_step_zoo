import torch.nn as nn
from nflows.transforms.base import CompositeTransform, MultiscaleCompositeTransform
from nflows.transforms.reshape import SqueezeTransform


from . import TwoStepDensityEstimator, GaussianVAE, AdversarialVariationalBayes
from .networks import MLP, CNN, T_CNN, GaussianMixtureLSTM
from .networks import SimpleFlowTransform
from .generalized_autoencoder import AutoEncoder, WassersteinAutoEncoder, BiGAN
from .density_estimator import NormalizingFlow, EnergyBasedModel, GaussianMixtureLSTMModel


activation_map = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "swish": nn.SiLU
}
_DEFAULT_ACTIVATION = "relu"


def get_two_step_module(gae_cfg, de_cfg, shared_cfg):
    gae_module = get_single_module(gae_cfg, **shared_cfg)

    # HACK: Allows specification of inferred `data_dim` using kwargs in single_main
    shared_cfg_copy = shared_cfg.copy()
    shared_cfg_copy["data_dim"] = de_cfg["data_dim"]
    shared_cfg_copy["data_shape"] = (de_cfg["data_dim"],)

    de_module = get_single_module(de_cfg, **shared_cfg_copy)

    two_step_module = TwoStepDensityEstimator(
        generalized_autoencoder=gae_module,
        density_estimator=de_module
    )

    return two_step_module


def get_single_module(cfg, **kwargs):
    cfg["data_dim"] = kwargs.get("data_dim", None)
    cfg["data_shape"] = kwargs.get("data_shape", None)

    model_to_module_map = {
        "vae": get_vae_module,
        "avb": get_avb_module,
        "ae": get_ae_module,
        "wae": get_wae_module,
        "bigan": get_bigan_module,
        "flow": get_flow_module,
        "ebm": get_ebm_module,
        "arm": get_arm_module,
    }
    module = model_to_module_map[cfg["model"]](cfg)

    lr_scheduler_args = ["max_epochs", "train_dataset_size", "train_batch_size"]
    for arg in lr_scheduler_args:
        if arg not in cfg:
            cfg[arg] = kwargs[arg]

    module.set_optimizer(cfg)

    return module


def get_vae_module(cfg):
    encoder, decoder = get_encoder_decoder(cfg)

    return GaussianVAE(
        latent_dim=cfg["latent_dim"],
        encoder=encoder,
        decoder=decoder,
        **get_data_transform_kwargs(cfg)
    )


def get_avb_module(cfg):
    encoder, decoder = get_encoder_decoder(cfg)
    discriminator = get_discriminator(cfg)

    return AdversarialVariationalBayes(
        latent_dim=cfg["latent_dim"],
        noise_dim=cfg["noise_dim"],
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        input_sigma=cfg["input_sigma"],
        prior_sigma=cfg["prior_sigma"],
        cnn=True if cfg["encoder_net"] == "cnn" else False,
        **get_data_transform_kwargs(cfg)
    )


def get_ae_module(cfg):
    encoder, decoder = get_encoder_decoder(cfg)

    return AutoEncoder(
        latent_dim=cfg["latent_dim"],
        encoder=encoder,
        decoder=decoder,
        **get_data_transform_kwargs(cfg)
    )


def get_wae_module(cfg):
    encoder, decoder = get_encoder_decoder(cfg)
    discriminator = get_discriminator(cfg)

    return WassersteinAutoEncoder(
        latent_dim=cfg["latent_dim"],
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        _lambda=cfg["_lambda"],
        sigma=cfg["sigma"],
        **get_data_transform_kwargs(cfg)
    )


def get_bigan_module(cfg):
    encoder, decoder = get_encoder_decoder(cfg)
    discriminator = get_discriminator(cfg)

    return BiGAN(
        latent_dim=cfg["latent_dim"],
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        wasserstein=cfg.get("wasserstein", True),
        clamp=cfg.get("clamp", 0.01),
        gradient_penalty=cfg.get("gradient_penalty", True),
        _lambda=cfg.get("lambda", 10),
        num_discriminator_steps=cfg.get("num_discriminator_steps", 5),
        recon_weight=cfg.get("recon_weight", 1.0),
        **get_data_transform_kwargs(cfg)
    )


def get_flow_module(cfg):
    if cfg["transform"] == "simple_nsf":
        transform = SimpleFlowTransform(
            features_for_mask=cfg["data_dim"],
            hidden_features=cfg["hidden_units"],
            num_layers=cfg["num_layers"],
            num_blocks_per_layer=cfg["num_blocks_per_layer"]
        )

    elif cfg["transform"] == "multiscale":
        transform = MultiscaleCompositeTransform(num_transforms=2)

        post_squeeze_dim = cfg["data_shape"][1]//2      # NOTE: Assumes square img
        post_squeeze_channels = 4*cfg["data_shape"][0]
        post_squeeze_shape = (post_squeeze_channels, post_squeeze_dim, post_squeeze_dim)

        pre_split_nsf_transform = SimpleFlowTransform(
            features_for_mask=post_squeeze_channels,
            hidden_features=cfg["hidden_units"],
            num_layers=cfg["num_layers"],
            num_blocks_per_layer=cfg["num_blocks_per_layer"],
            net="cnn"
        )
        pre_split_transform = CompositeTransform(
            transforms=(SqueezeTransform(), pre_split_nsf_transform)
        )

        post_split_shape = transform.add_transform(
            pre_split_transform,
            post_squeeze_shape
        )

        post_split_transform = SimpleFlowTransform(
            features_for_mask=post_split_shape[0],
            hidden_features=cfg["hidden_units"],
            num_layers=cfg["num_layers"],
            num_blocks_per_layer=cfg["num_blocks_per_layer"],
            net="cnn"
        )
        transform.add_transform(
            post_split_transform,
            post_split_shape
        )

    else:
        raise NotImplementedError(f"Transform {cfg['transform']} not implemented")

    if cfg.get("base_distribution", None) == None:
        base_distribution = None
    else:
        raise NotImplementedError(f"Base distribution {cfg['base_distribution']} not implemented")

    return NormalizingFlow(
        dim=cfg["data_dim"],
        transform=transform,
        base_distribution=base_distribution,
        **get_data_transform_kwargs(cfg)
    )


def get_ebm_module(cfg):
    _DEFAULT_EBM_ACTIVATION = "swish"

    if cfg["net"] == "mlp":
        energy_func = MLP(
            input_dim=cfg["data_dim"],
            hidden_dims=cfg["energy_func_hidden_dims"],
            output_dim=1,
            activation=activation_map[cfg.get("energy_func_activation", _DEFAULT_EBM_ACTIVATION)],
            spectral_norm=cfg.get("spectral_norm", False),
        )

    elif cfg["net"] == "cnn":
        energy_func = CNN(
            input_channels=cfg["data_shape"][0],
            hidden_channels_list=cfg["energy_func_hidden_channels"],
            output_dim=1,
            kernel_size=cfg["energy_func_kernel_size"],
            stride=cfg["energy_func_stride"],
            image_height=cfg["data_shape"][1],
            activation=activation_map[cfg.get("energy_func_activation", _DEFAULT_EBM_ACTIVATION)],
            spectral_norm=cfg.get("spectral_norm", False),
        )

    else:
        raise ValueError(f"Unknown network type {cfg['net']} for EBM")

    if cfg.get("flatten", False):
        x_shape = (cfg["data_dim"],)
    else:
        x_shape = cfg["data_shape"]

    return EnergyBasedModel(
        energy_func=energy_func,
        x_shape=x_shape,
        max_length_buffer=cfg["max_length_buffer"],
        x_lims=cfg["x_lims"],
        ld_steps=cfg["ld_steps"],
        ld_step_size=cfg["ld_step_size"],
        ld_eps_new=cfg["ld_eps_new"],
        ld_sigma=cfg["ld_sigma"],
        ld_grad_clamp=cfg["ld_grad_clamp"],
        loss_alpha=cfg["loss_alpha"],
        **get_data_transform_kwargs(cfg)
    )


def get_arm_module(cfg):
    ar_network = GaussianMixtureLSTM(
        input_size=(1 if len(cfg["data_shape"])==1 else cfg["data_shape"][0]),
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        k_mixture=cfg["k_mixture"]
    )

    image_height = None if len(cfg["data_shape"]) == 1 else cfg["data_shape"][1]
    image_length = (
        cfg["data_shape"][0] if len(cfg["data_shape"]) == 1
        else cfg["data_shape"][1]*cfg["data_shape"][2]
    )
    return GaussianMixtureLSTMModel(
        ar_network=ar_network,
        image_height=image_height,
        input_length=image_length,
        **get_data_transform_kwargs(cfg)
    )


def get_encoder_decoder(cfg):
    model = cfg["model"]

    if model == "vae":
        encoder_output_dim = 2*cfg["latent_dim"]
        encoder_output_split_sizes = [cfg["latent_dim"], cfg["latent_dim"]]
    else:
        encoder_output_dim = cfg["latent_dim"]
        encoder_output_split_sizes = None

    if cfg["encoder_net"] == "mlp":
        encoder = MLP(
            input_dim=cfg["data_dim"]+cfg.get("noise_dim", 0),
            hidden_dims=cfg["encoder_hidden_dims"],
            output_dim=encoder_output_dim,
            activation=activation_map[cfg.get("encoder_activation", _DEFAULT_ACTIVATION)],
            output_split_sizes=encoder_output_split_sizes
        )

    elif cfg["encoder_net"] == "cnn":
        encoder = CNN(
            input_channels=cfg["data_shape"][0],
            hidden_channels_list=cfg["encoder_hidden_channels"],
            output_dim=encoder_output_dim,
            kernel_size=cfg["encoder_kernel_size"],
            stride=cfg["encoder_stride"],
            image_height=cfg["data_shape"][1],
            activation=activation_map[cfg.get("encoder_activation", _DEFAULT_ACTIVATION)],
            output_split_sizes=encoder_output_split_sizes,
            noise_dim=cfg.get("noise_dim", 0)
        )

    else:
        raise ValueError(f"Unknown encoder network type {cfg['encoder_net']}")

    if cfg["decoder_net"] == "mlp":
        if model in ["avb", "vae"]:
            decoder_sigma_dim = 1 if cfg["single_sigma"] else cfg["data_dim"]
            decoder_output_dim = cfg["data_dim"] + decoder_sigma_dim
            decoder_output_split_sizes = [cfg["data_dim"], decoder_sigma_dim]
        else:
            decoder_output_dim = cfg["data_dim"]
            decoder_output_split_sizes = None

        decoder = MLP(
            input_dim=cfg["latent_dim"],
            hidden_dims=cfg["decoder_hidden_dims"],
            output_dim=decoder_output_dim,
            activation=activation_map[cfg.get("decoder_activation", _DEFAULT_ACTIVATION)],
            output_split_sizes=decoder_output_split_sizes
        )

    elif cfg["decoder_net"] == "cnn":
        if model in ["avb", "vae"]:
            decoder_sigma_dim = 1 if cfg["single_sigma"] else cfg["data_dim"]
            decoder_output_channels = cfg["data_shape"][0] + decoder_sigma_dim
            decoder_output_split_sizes = [cfg["data_shape"][0], decoder_sigma_dim]
        else:
            decoder_output_channels = cfg["data_shape"][0]
            decoder_output_split_sizes = None

        decoder = T_CNN(
            input_dim=cfg["latent_dim"],
            hidden_channels_list=cfg["decoder_hidden_channels"],
            output_channels=decoder_output_channels,
            kernel_size=cfg["decoder_kernel_size"],
            stride=cfg["decoder_stride"],
            image_height=cfg["data_shape"][1],
            activation=activation_map[cfg.get("decoder_activation", _DEFAULT_ACTIVATION)],
            output_split_sizes=decoder_output_split_sizes,
            single_sigma=cfg.get("single_sigma", False)
        )
    else:
        raise ValueError(f"Unknown decoder network type {cfg['decoder_net']}")

    return encoder, decoder


def get_discriminator(cfg):
    extra_dims = cfg["data_dim"] if cfg["model"] in ["avb", "bigan"] else 0

    return MLP(
        input_dim=cfg["latent_dim"]+extra_dims,
        hidden_dims=cfg["discriminator_hidden_dims"],
        output_dim=1,
        activation=activation_map[cfg.get("discriminator_activation", _DEFAULT_ACTIVATION)]
    )


def get_data_transform_kwargs(cfg):
    return {
        "flatten": cfg.get("flatten", False),
        "data_shape": cfg.get("data_shape", None),
        "denoising_sigma": cfg.get("denoising_sigma", None),
        "dequantize": cfg.get("dequantize", False),
        "scale_data": cfg.get("scale_data", False),
        "whitening_transform": cfg.get("whitening_transform", False),
        "logit_transform": cfg.get("logit_transform", False),
        "clamp_samples": cfg.get("clamp_samples", False),
    }
