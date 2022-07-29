import math
import torch
from torch import nn


class BaseNetworkClass(nn.Module):
    def __init__(self, output_split_sizes):
        super().__init__()

        self.output_split_sizes = output_split_sizes

    def _get_correct_nn_output_format(self, nn_output, split_dim):
        if self.output_split_sizes is not None:
            return torch.split(nn_output, self.output_split_sizes, dim=split_dim)
        else:
            return nn_output

    def _apply_spectral_norm(self):
        for module in self.modules():
            if "weight" in module._parameters:
                nn.utils.spectral_norm(module)

    def forward(self, x):
        raise NotImplementedError("Define in child classes.")


class MLP(BaseNetworkClass):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            output_dim,
            activation,
            output_split_sizes=None,
            spectral_norm=False
    ):
        super().__init__(output_split_sizes)

        layers = []
        prev_layer_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features=prev_layer_dim, out_features=hidden_dim))
            layers.append(activation())
            prev_layer_dim = hidden_dim

        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))
        self.net = nn.Sequential(*layers)

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x):
        return self._get_correct_nn_output_format(self.net(x), split_dim=-1)


class BaseCNNClass(BaseNetworkClass):
    def __init__(
            self,
            hidden_channels_list,
            stride,
            kernel_size,
            output_split_sizes=None
    ):
        super().__init__(output_split_sizes)
        self.stride = self._get_stride_or_kernel(stride, hidden_channels_list)
        self.kernel_size = self._get_stride_or_kernel(kernel_size, hidden_channels_list)

    def _get_stride_or_kernel(self, s_or_k, hidden_channels_list):
        if type(s_or_k) not in [list, tuple]:
            return [s_or_k for _ in hidden_channels_list]
        else:
            assert len(s_or_k) == len(hidden_channels_list), \
                "Mismatch between stride/kernels provided and number of hidden channels."
            return s_or_k


class CNN(BaseCNNClass):
    def __init__(
            self,
            input_channels,
            hidden_channels_list,
            output_dim,
            kernel_size,
            stride,
            image_height,
            activation,
            output_split_sizes=None,
            spectral_norm=False,
            noise_dim=0
    ):
        super().__init__(hidden_channels_list, stride, kernel_size, output_split_sizes)

        cnn_layers = []
        prev_channels = input_channels
        for hidden_channels, k, s in zip(hidden_channels_list, self.kernel_size, self.stride):
            cnn_layers.append(nn.Conv2d(prev_channels, hidden_channels, k, s))
            cnn_layers.append(activation())
            prev_channels = hidden_channels

            # NOTE: Assumes square image
            image_height = self._get_new_image_height(image_height, k, s)
        self.cnn_layers = nn.ModuleList(cnn_layers)

        self.fc_layer = nn.Linear(
            in_features=prev_channels*image_height**2+noise_dim,
            out_features=output_dim
        )

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x, eps=None):
        for layer in self.cnn_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)

        net_in = torch.cat((x, eps), dim=1) if eps is not None else x
        net_out = self.fc_layer(net_in)

        return self._get_correct_nn_output_format(net_out, split_dim=1)

    def _get_new_image_height(self, height, kernel, stride):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv2d.html
        # Assume dilation = 1, padding = 0
        return math.floor((height - kernel)/stride + 1)


class T_CNN(BaseCNNClass):
    def __init__(
            self,
            input_dim,
            hidden_channels_list,
            output_channels,
            kernel_size,
            stride,
            image_height,
            activation,
            output_split_sizes=None,
            spectral_norm=False,
            single_sigma=False,
    ):
        super().__init__(hidden_channels_list, stride, kernel_size, output_split_sizes)

        self.single_sigma = single_sigma

        if self.single_sigma:
            # NOTE: In the MLP above, the single_sigma case can be handled by correctly
            #       specifying output_split_sizes. However, here the first output of the
            #       network will be of image shape, which more strongly contrasts with the
            #       desired sigma output of shape (batch_size, 1). We need the additional
            #       linear layer to project the image-like output to a scalar.
            self.sigma_output_layer = nn.Linear(output_split_sizes[-1]*image_height**2, 1)

        output_paddings = []
        for _, k, s in zip(hidden_channels_list, self.kernel_size[::-1], self.stride[::-1]):
            # First need to infer the appropriate number of outputs for the first linear layer
            image_height, output_padding = self._get_new_image_height_and_output_padding(
                image_height, k, s
            )
            output_paddings.append(output_padding)
        output_paddings = output_paddings[::-1]

        self.fc_layer = nn.Linear(input_dim, hidden_channels_list[0]*image_height**2)
        self.post_fc_shape = (hidden_channels_list[0], image_height, image_height)

        t_cnn_layers = [activation()]
        prev_channels = hidden_channels_list[0]
        for hidden_channels, k, s, op in zip(
            hidden_channels_list[1:], self.kernel_size[:-1], self.stride[:-1], output_paddings[:-1]
        ):
            t_cnn_layers.append(
                nn.ConvTranspose2d(prev_channels, hidden_channels, k, s, output_padding=op)
            )
            t_cnn_layers.append(activation())

            prev_channels = hidden_channels

        t_cnn_layers.append(
            nn.ConvTranspose2d(
                prev_channels, output_channels, self.kernel_size[-1], self.stride[-1],
                output_padding=output_paddings[-1]
            )
        )
        self.t_cnn_layers = nn.ModuleList(t_cnn_layers)

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x):
        x = self.fc_layer(x).reshape(-1, *self.post_fc_shape)

        for layer in self.t_cnn_layers:
            x = layer(x)
        net_output = self._get_correct_nn_output_format(x, split_dim=1)

        if self.single_sigma:
            mu, log_sigma_unprocessed = net_output
            log_sigma = self.sigma_output_layer(log_sigma_unprocessed.flatten(start_dim=1))
            return mu, log_sigma.view(-1, 1, 1, 1)
        else:
            return net_output

    def _get_new_image_height_and_output_padding(self, height, kernel, stride):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.ConvTranspose2d.html
        # Assume dilation = 1, padding = 0
        output_padding = (height - kernel) % stride
        height = (height - kernel - output_padding) // stride + 1

        return height, output_padding


class GaussianMixtureLSTM(BaseNetworkClass):
    def __init__(self, input_size,  hidden_size, num_layers, k_mixture):
        output_split_sizes = [k_mixture, k_mixture * input_size, k_mixture * input_size]
        super().__init__(output_split_sizes)
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=k_mixture*(2*input_size+1))
        self.hidden_size = hidden_size
        self.k = k_mixture

    def forward(self, x, return_h_c=False, h_c=None, not_sampling=True):
        if not_sampling:
            x = torch.flatten(x, start_dim=2)
            x = torch.permute(x, (0, 2, 1))  # move "channels" to last axis
            x = x[:, :-1]  # last coordinate is never used as input
        if h_c is None:
            out, h_c = self.rnn(x)
        else:
            out, h_c = self.rnn(x, h_c)
        if not_sampling:
            out = torch.cat((torch.zeros((out.shape[0], 1, self.hidden_size)).to(out.device), out), dim=1)
        out = self.linear(out)
        weights, mus, sigmas = self.split_transform_and_reshape(out)  # NOTE: weights contains logits
        if return_h_c:
            return weights, mus, sigmas, h_c
        else:
            return weights, mus, sigmas

    def split_transform_and_reshape(self, out):
        weights, mus, sigmas = self._get_correct_nn_output_format(out, split_dim=-1)
        mus = torch.reshape(mus, (mus.shape[0], mus.shape[1], self.k, -1))
        sigmas = torch.reshape(torch.exp(sigmas), (sigmas.shape[0], sigmas.shape[1], self.k, -1))
        return weights, mus, sigmas
