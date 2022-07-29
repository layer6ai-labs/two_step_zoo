import torch
import torch.nn.functional as F

from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.conv import OneByOneConvolution


class SimpleFlowTransform(CompositeTransform):
    """Simple flow transform acting on flat or image data depending on `net`"""

    def __init__(
        self,
        features_for_mask, # flat data: number of features; image data: number of channels
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        include_linear=True,
        num_bins=8,
        tail_bound=1.0,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        coupling_constructor=PiecewiseRationalQuadraticCouplingTransform,
        net="mlp"
    ):
        mask = torch.ones(features_for_mask)
        mask[::2] = -1

        self.model_type = net

        def create_resnet(in_features, out_features):
            if net == "cnn":
                return nets.ConvResidualNet(
                    in_features,
                    out_features,
                    hidden_channels=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            else:
                return nets.ResidualNet(
                    in_features,
                    out_features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )

        layers = []
        for _ in range(num_layers):
            coupling_transform = coupling_constructor(
                mask=mask,
                transform_net_create_fn=create_resnet,
                tails="linear",
                num_bins=num_bins,
                tail_bound=tail_bound,
            )
            layers.append(coupling_transform)
            mask *= -1

            if include_linear:

                if self.model_type == "cnn":
                    linear_transform = CompositeTransform([
                        OneByOneConvolution(features_for_mask, identity_init=True)
                    ])
                else:
                    linear_transform = CompositeTransform([
                        RandomPermutation(features=features_for_mask),
                        LULinear(features_for_mask, identity_init=True)])
                layers.append(linear_transform)

        super().__init__(layers)
