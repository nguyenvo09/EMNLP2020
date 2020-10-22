from torch import nn
import torch


class PACRRPlaneMaxPooling(nn.Module):
    """
    head feature extractor
    """
    def __init__(self, num_conv_layers: int, input_channels: int, filters_count: int, ns: int):
        super(PACRRPlaneMaxPooling, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.n_s = ns
        self.n_conv_layers = num_conv_layers
        self.filters_count = filters_count
        for i in range(num_conv_layers):
            # expected input shape is (B, 1, L, R) => (B, filters, L, R)
            conv = nn.Sequential(
                # The simplest way to make same padding for (i + 2)-grams. We pad bottom and right only
                # https://pytorch.org/docs/stable/_modules/torch/nn/modules/padding.html
                nn.ConstantPad2d((0, i, 0, i), 0),
                nn.Conv2d(
                    in_channels=input_channels,  # the depth is 1 (for sim)
                    out_channels=filters_count,  # the number of filters for each kernel_size
                    kernel_size=i + 1,  # sqare windows
                    # kernel size starts from 1 since if kerner-size = 1, it is unigram = original matrix
                    stride=(1, 1),
                ),
                nn.Tanh()
            )
            self.conv_layers.append(conv)
            self.last_in_channels = num_conv_layers
            self.L_new = filters_count
            self.R_new = ns

    def forward(self, tensor: torch.Tensor):
        """x \in [B, C, L, R]"""
        B, C, L, R = tensor.size()
        Conv_Pool_Outs = []  # add unigram signal
        full_indices = []
        for conv in self.conv_layers:  # looped l_g times or conv_layers times l_g == conv_layers
            x = conv(tensor)  # (B, n_filters, L, R)
            y = x.view(B, self.filters_count, -1)  # (B, n_filters, L * R)
            top_toks, _ = y.topk(self.n_s, dim=-1)  # (B, n_filters, n_s)
            Conv_Pool_Outs.append(top_toks)
        assert len(Conv_Pool_Outs) == self.n_conv_layers
        phi = torch.stack(Conv_Pool_Outs, dim=-1)  # B, n_filters, n_s, n_conv_layers,
        phi = phi.permute(0, 3, 2, 1)  # B, n_conv_layers, n_filters, n_s
        return phi
