#%%
import torch
import torch.nn as nn
from collections import OrderedDict


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)

    def forward(self, x):
        return self.conv(x)

class L2N(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        inputs = inputs / torch.norm(inputs, p=2, dim=1, keepdim=True)
        return inputs


class ProbabilisitcEncoder(nn.Module):
    def __init__(self, encoder_id) -> None:
        super().__init__()
        self.encoder_id = encoder_id
        self.var_init = 1e-3
        self.encoder = nn.Sequential(OrderedDict([
                ('conv1', Conv(1, 16, 6, 2)),
                ('ac1', nn.SiLU()),
                ('conv2', Conv(16, 32, 6, 2)),
                ('ac2', nn.SiLU())
                ]))
        self.mu_branch = nn.Sequential(OrderedDict([
            ('conv1_mu', Conv(32, 32, 6, 2)),
            ('ac1_mu', nn.SiLU()),
            ('conv2_mu', Conv(32, 32, 6, 2)),
            ('ac2_mu', L2N()),]))
        self.sigma_branch = nn.Sequential(
            OrderedDict([
                ('conv1_sigma', Conv(32, 32, 6, 2)),
                ('ac1_sigma', nn.SiLU()),
                ('conv2_sigma', Conv(32, 1, 6, 2)),
                ('sf_sigma', nn.Softplus()),]))
        init_method = 0
        if init_method == 0:
            self.sigma_branch[0].conv.weight.data.zero_()
            self.sigma_branch[0].conv.bias.data.copy_(torch.log(torch.tensor(self.var_init)))
        elif init_method == 1:
            self.sigma_branch[0].conv.weight.data.zero_()
            torch.nn.init.uniform_(self.sigma_branch[0].conv.bias,
                                   a=torch.log(torch.tensor(self.var_init)),    # (-6.9, 6.9)
                                   b=-torch.log(torch.tensor(self.var_init)))
        elif init_method == 2:
            self.sigma_branch[0].conv.weight.data.zero_()
            torch.nn.init.uniform_(self.sigma_branch[0].conv.bias,
                                   a=-self.var_init,    # (-1e-3, 1e-3)
                                   b=self.var_init)
        elif init_method == 3:
            self.sigma_branch[0].conv.weight.data.zero_()
            torch.nn.init.uniform_(
                self.sigma_branch[0].conv.bias,
                a=-0.5,  # (-1e-3, 1e-3)
                b=0.5)

    def forward(self, inputs):  # (B, C, H, W)
        fm = self.encoder(inputs)  # (B, C, H, W)
        mu = self.mu_branch(fm)  # (B, C, H, W)
        sigma = self.sigma_branch(fm)  # (B, C, H, W)
        prob_feat = (mu, sigma)
        return prob_feat


class Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.thermal_encoder = ProbabilisitcEncoder('thermal')
        self.radar_encoder = ProbabilisitcEncoder('radar')

    def forward(self, inputs):
        tml_img, rad_img = inputs[:, 0, :, :,].unsqueeze(1), inputs[:, 2, :, :,].unsqueeze(1)
        tml_fm_mu, tml_fm_sigma = self.thermal_encoder(tml_img) # outputs: mu, sigma^2
        rad_fm_mu, rad_fm_sigma = self.radar_encoder(rad_img)
        mu_fused = (tml_fm_mu * rad_fm_sigma ** (0.5) + rad_fm_mu * tml_fm_sigma ** (0.5)) / \
             (tml_fm_sigma ** (0.5) * rad_fm_sigma ** (0.5) + 1e-6)

        return {
            'mu_fused': mu_fused,
            'mu_tml': tml_fm_mu,
            'sigma_tml': tml_fm_sigma,
            'mu_rad': rad_fm_mu,
            'sigma_rad': rad_fm_sigma,}


if __name__ == '__main__':
    B, C, H, W = 1, 1, 640, 640
    device=torch.device('cuda')

    tml_img, rad_img = torch.rand((B, C, H, W), device=device), torch.rand((B, C, H, W), device=device)                    # batch_size, channel, h, w
    inputs = torch.cat((tml_img, rad_img), dim=1)
    fusion_net = Fusion()

    from torchsummary import summary
    summary(fusion_net.to(device), input_size=(2*C, H, W))

    out = fusion_net(inputs)
    print(f'Fused feature map shape: {out.shape}')
