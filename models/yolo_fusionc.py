# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
from os.path import join, exists
import torch.nn.functional as F
import torch.nn as nn
import pickle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization, x2img
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once_multifusion(self, x, profile=False, visualize=False): # x:[B,C,H,W]
        imgs_tml = x[:, 2, :, :].unsqueeze(1).repeat(1, 3, 1, 1) # [radar, thermal, thermal]  ([32, 3, 640, 640])
        imgs_aux = x[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
        x = deepcopy(imgs_tml)
        aux_x = deepcopy(imgs_aux)
        y, dt = [], []  # outputs

        for i in range(5):
            x = self.model[i](x)
            aux_x = self.aux_encoder[i](aux_x)
            x += aux_x

        y = [None, None, None, None, x]
        for i, m in enumerate(self.model[5:]):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x

    def _forward_once(self, x, profile=False, visualize=False): # x:[B,C,H,W]
        '''
        after
        for datasets:           0-radar,    1-thermal,  2-thermal
        for datasets_lepton:    0-radar,      1-rcs,    2-thermal
        '''

        # # general
        imgs_tml = x[:, 2, :, :].unsqueeze(1).repeat(1, 3, 1, 1)  # [thermal, thermal, thermal]  ([32, 3, 640, 640])
        imgs_aux = x[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)  # [radar, radar, radar]

        # # lepton rcs
        # imgs_tml = torch.cat((x[:, 2, :, :].unsqueeze(1), x[:, 2, :, :].unsqueeze(1), x[:, 0, :, :].unsqueeze(1)), dim=1)  # [thermal, thermal, radar]
        # imgs_aux = x[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)  # [rcs, rcs, rcs]

        x = deepcopy(imgs_tml)
        aux_x = deepcopy(imgs_aux)
        # if len(x) > 1:
        #     x, aux_x = x
        #     img_tml, img_rad = deepcopy(x), deepcopy(aux_x)
        y, dt = [], []  # outputs

        xs, aux_xs = [], []
        for _ in range(self.dropout_T):
            xs.append(self.model[:5](x))    # ([32, 128, 80, 80])
            aux_xs.append(self.aux_encoder(aux_x))  # ([32, 128, 80, 80])
        xs_m, xs_v = torch.stack(xs, 0).mean(0), torch.stack(xs, 0).var(0)  # ([1, 128, 32, 32]) <= ([10, 1, 128, 32, 32]).mean(0)
        aux_xs_m, aux_xs_v = torch.stack(aux_xs, 0).mean(0), torch.stack(aux_xs, 0).var(0)  # ([32, 128, 80, 80]), # ([32, 128, 80, 80])

        # # sanity check
        # if self.dropout_T > 1 and xs_v.max() == 0 and xs_v.min() == 0:
        #     if xs_m.max() == 0 and xs_m.min() == 0:
        #         pass
        #     elif not self.training:
        #         pass
        #     else:
        #         print('Warning: zero variance with T>1')

        if self.fusion_method == 'main_only':
            x = xs_m

        elif self.fusion_method == 'vanilla_addition':
            x = xs_m + aux_xs_m

        elif self.fusion_method == 'vanilla_product':
            x = xs_m * aux_xs_m

        elif self.fusion_method == 'attention_1':
            att_map = F.softmax(self.att_conv(aux_xs_m), dim=1)
            x = xs_m * att_map

        elif self.fusion_method == 'attention_2':
            att_map = F.softmax(self.att_conv(aux_xs_m), dim=1)  # ([32, 128, 20, 20])
            query_map = self.query_conv(xs_m)
            x = query_map * att_map

        elif self.fusion_method == 'spatial_variance_self_guidence':    # att3
            b, c, h, w = aux_xs_m.shape
            main_map = F.softmax(torch.sigmoid(xs_v.reshape(b, c, -1)), dim=-1).reshape(b, c, h, w)
            main_weighted = main_map * xs_m

            aux_map = F.softmax(torch.sigmoid(aux_xs_v.reshape(b, c, -1)), dim=-1).reshape(b, c, h, w)
            aux_weighted = aux_map * aux_xs_m

            x = main_weighted + aux_weighted
            # x = self.ac(self.bn(x))

        elif self.fusion_method == 'channel_variance_cross_guidence':
            b, c, h, w = aux_xs_m.shape
            main_map = F.softmax(torch.sigmoid(xs_v.reshape(b, c, -1)), dim=1).reshape(b, c, h, w)
            aux_map = F.softmax(torch.sigmoid(aux_xs_v.reshape(b, c, -1)), dim=1).reshape(b, c, h, w)

            main_weighted = aux_map * xs_m
            aux_weighted = main_map * aux_xs_m

            x = main_weighted + aux_weighted

        elif self.fusion_method == 'channel_variance_self_guidence':
            b, c, h, w = aux_xs_m.shape
            main_map = F.softmax(torch.sigmoid(xs_v.reshape(b, c, -1)), dim=1).reshape(b, c, h, w)
            main_weighted = main_map * xs_m

            aux_map = F.softmax(torch.sigmoid(aux_xs_v.reshape(b, c, -1)), dim=1).reshape(b, c, h, w)
            aux_weighted = aux_map * aux_xs_m

            x = main_weighted + aux_weighted

        elif self.fusion_method == 'spatial_variance_cross_guidence':
            b, c, h, w = aux_xs_m.shape
            x = xs_m + aux_xs_m
            x = x * F.softmax(torch.sigmoid(aux_xs_v.reshape(b, c, -1)), dim=-1).reshape(b, c, h, w)  # ([32, 128, 80, 80])

        elif self.fusion_method == 'vallina_variance_1':
            v1 = xs_v.view(32, -1)
            index = torch.randperm(v1.shape[-1]).to(v1.device)
            v2 = v1[:, index]
            v1_ = v1 / (v1 + v2)
            v2_ = v2 / (v1 + v2)
            print(v1_.min().item(), v1_.mean().item(), v1_.max().item())
            print(v2_.min().item(), v2_.mean().item(), v2_.max().item())

            v1 = xs_v.view(32, -1)
            v2 = aux_xs_v.view(32, -1)
            v1_ = v1 / (v1 + v2 + 1)
            v2_ = v2 / (v1 + v2 + 1)
            print(v1_.min().item(), v1_.mean().item(), v1_.max().item())
            print(v2_.min().item(), v2_.mean().item(), v2_.max().item())

            xs_v = self.bn1(xs_v)
            aux_xs_v = self.bn2(aux_xs_v)
            v1 = xs_v / (xs_v + aux_xs_v + 1e-5)
            v2 = aux_xs_v / (xs_v + aux_xs_v)
            u1 = xs_m
            u2 = aux_xs_m
            x = v1*u1 + v2*u2

        elif self.fusion_method == 'vallina_variance_2':
            v1 = xs_v / (xs_v + aux_xs_v)
            v2 = aux_xs_v / (xs_v + aux_xs_v)
            u1 = xs_m
            u2 = aux_xs_m
            x = v2 * u1 + v1 * u2

        else:
            print(f'{self.fusion_method} undefined.')
            raise('undefined fusion method')

        # debug area
        if exists("tmp/counter.npy"):
            def wrap(filename):
                batch_i = np.load("tmp/counter.npy")
                return join('tmp', f"{batch_i}_{filename}")
            b_index, c_index = 0, 0
            x2img(imgs_tml[b_index, c_index, :, :], wrap('imgs_tml.png'))  # input
            x2img(imgs_aux[b_index, c_index, :, :], wrap('imgs_aux.png'))
            x2img(xs_m[b_index, c_index, :, :], wrap('xs_m.png'), cmap='gray')  # feature
            x2img(aux_xs_m[b_index, c_index, :, :], wrap('aux_xs_m.png'), cmap='gray')
            x2img(xs_v[b_index, c_index, :, :], wrap('xs_v.png'))  # feature variance
            x2img(aux_xs_v[b_index, c_index, :, :], wrap('aux_xs_v.png'))
            if self.fusion_method == "spatial_variance_self_guidence":
                x2img(main_map[b_index, c_index, :, :], wrap('main_map.png'))  # rectified feature variance map
                x2img(aux_map[b_index, c_index, :, :], wrap('aux_map.png'))
                x2img(main_weighted[b_index, c_index, :, :], wrap('main_weighted.png'))  # rectified feature variance map
                x2img(aux_weighted[b_index, c_index, :, :], wrap('aux_weighted.png'))
            elif self.fusion_method == 'attention_2':
                x2img(query_map[b_index, c_index, :, :], wrap('query_map.png'))  # rectified feature variance map
                x2img(att_map[b_index, c_index, :, :], wrap('att_map.png'))
                # with open('0_att.pickle', 'wb') as handle:
                #     pickle.dump(imgs_tml[b_index, c_index, :, :], handle, protocol=pickle.HIGHEST_PROTOCOL)
                #     pickle.dump(xs_m[b_index, c_index, :, :], handle, protocol=pickle.HIGHEST_PROTOCOL)
                #     pickle.dump(query_map[b_index, c_index, :, :], handle, protocol=pickle.HIGHEST_PROTOCOL)
                #     pickle.dump(imgs_aux[b_index, c_index, :, :], handle, protocol=pickle.HIGHEST_PROTOCOL)
                #     pickle.dump(aux_xs_m[b_index, c_index, :, :], handle, protocol=pickle.HIGHEST_PROTOCOL)
                #     pickle.dump(att_map[b_index, c_index, :, :], handle, protocol=pickle.HIGHEST_PROTOCOL)

            x2img(x[b_index, c_index, :, :], wrap('x.png'))  # output




        # x2img(torch.sigmoid(xs_v[b_index, c_index, :, :]), wrap('xs_v_sigmoid.png'))
        # x2img(F.softmax(torch.sigmoid(xs_v[b_index, c_index, :, :].reshape(-1)), dim=0).reshape(80, 80), 'xs_v_sigmoid_softmax.png')
        # x2img(xs_m[b_index, c_index, :, :], 'xs_m.png')
        # x2img(xs_m[b_index, c_index, :, :] * torch.sigmoid(aux_xs_v[b_index, c_index, :, :]), 'xs_m-aux_xs_v_sigmoid.png')
        # x2img(xs_m[b_index, c_index, :, :] * F.softmax(torch.sigmoid(aux_xs_v[b_index, c_index, :, :].reshape(-1)), dim=0).reshape(80, 80), 'xs_m-aux_xs_v_sigmoid_softmax.png')
        # x2img(imgs_aux[b_index, c_index, :, :], 'aux_xs_img.png')
        # x2img(aux_xs_v[b_index, c_index, :, :], 'aux_xs_v.png')
        # x2img(torch.sigmoid(aux_xs_v[b_index, c_index, :, :]), 'aux_xs_v_sigmoid.png')
        # # x2img(F.softmax(torch.sigmoid(aux_xs_v[b_index, c_index, :, :]), dim=0), 'aux_xs_v_sigmoid_softmax.png')
        # x2img(F.softmax(torch.sigmoid(aux_xs_v[b_index, c_index, :, :].reshape(-1)), dim=0).reshape(80, 80), 'aux_xs_v_sigmoid_softmax.png')

        if visualize:
            visualize_dir = join(visualize, 'main')
            if not exists(visualize_dir):
                os.makedirs(visualize_dir)
            visualize_aux_dir = join(visualize, 'aux')
            if not exists(visualize_aux_dir):
                os.makedirs(visualize_aux_dir)
            feature_visualization(xs_m, self.model[4].type, self.model[4].i, save_dir=visualize_dir)
            feature_visualization(xs_v, self.model[4].type + '_v', self.model[4].i, save_dir=visualize_dir)
            feature_visualization(aux_xs_m, self.aux_encoder[4].type, self.aux_encoder[4].i, save_dir=visualize_aux_dir)
            feature_visualization(aux_xs_v, self.aux_encoder[4].type + '_v', self.aux_encoder[4].i, save_dir=visualize_aux_dir)
            feature_visualization(imgs_tml, 'A', 0, save_dir=visualize_dir)
            feature_visualization(imgs_aux, 'A', 0, save_dir=visualize_aux_dir)

        y = [None, None, None, None, x]
        for i, m in enumerate(self.model[5:]):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            # if visualize:
            #     visualize_dir = join(visualize, 'main')
            #     if not exists(visualize_dir):
            #         os.makedirs(visualize_dir)
            #     visualize_aux_dir = join(visualize, 'aux')
            #     if not exists(visualize_aux_dir):
            #         os.makedirs(visualize_aux_dir)
            #     feature_visualization(x, m.type, m.i, save_dir=visualize_dir)
            #     feature_visualization(aux_x, self.aux_encoder[i].type, self.aux_encoder[i].i, save_dir=visualize_aux_dir)
            #     if m.i == 0:
            #         feature_visualization(imgs_tml, 'input', 0, save_dir=visualize_dir)
            #         feature_visualization(imgs_aux, 'input', 0, save_dir=visualize_aux_dir)

        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, opt=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name # 'yolov5s.yaml'
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.aux_encoder = deepcopy(self.model[:5])
        self.dropout_T = opt.dropout_T
        self.fusion_method = opt.fusion_method

        if self.fusion_method in ['attention_1']:
            self.att_conv = Conv(128, 128, 1, 1)
        elif self.fusion_method in ['attention_2']:
            self.att_conv = Conv(128, 128, 1, 1)
            self.query_conv = Conv(128, 128, 1, 1)
        elif self.fusion_method in ['vallina_variance_1', 'vallina_variance_2']:
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(128)
        elif self.fusion_method in ['spatial_variance_self_guidence', 'channel_variance_self_guidence', 'channel_variance_cross_guidence']:
            self.ac = nn.SiLU()
            self.bn = nn.BatchNorm2d(128)

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward    ([8,16,32])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None

        if self.fusion_method in ['multifusion']:
            return self._forward_once_multifusion(x, profile, visualize)  # single-scale inference, train
        else:
            return self._forward_once(x, profile, visualize)  # single-scale inference, train


    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.act = eval(act)  # redefine default activation, i.e. Conv.act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                VarConv, Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, VarC3, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, VarC3, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
