"""
Copy from https://github.com/chenyaofo/pytorch-cifar-models.
Then adding some codes for modular training
"""

import sys
import torch
import torch.nn as nn
import copy
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Union, List, Dict, Any, cast
from models.utils_v2 import MaskConvBN


cifar10_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
}

cifar100_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
}


class VGG(nn.Module):

    def __init__(
            self,
            features: nn.Module,
            num_classes: int = 10,
            init_weights: bool = True,
            n_last_conv_kernels:int=None,
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_last_conv_kernels, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self.module_head = None  # set the module head after instantiating the module and loading model params.

        self.num_classes = num_classes

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        x = self.module_head(x)
        return x

    def _initialize_weights(self) -> None:
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_masks(self):
        masks = [
            self.features[i].masks for i in range(len(self.features._modules)) if hasattr(self.features[i], 'masks')
        ]
        return masks


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, module_layer_masks=None, keep_generator=True) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    mask_layer_idx = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)  # The first conv layer will not be masked.
            if in_channels == 3:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(num_features=v))
                layers.append(nn.ReLU(inplace=True))
                first_norm_conv_pseudo_mask = torch.ones(v, dtype=module_layer_masks[0].dtype)
            else:
                pre_layer_mask = module_layer_masks[mask_layer_idx - 1] if mask_layer_idx > 0 else first_norm_conv_pseudo_mask
                cur_layer_mask = module_layer_masks[mask_layer_idx]
                mask_layer_idx += 1

                mask_conv_bn = MaskConvBN(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1,
                                          module_layer_masks=[pre_layer_mask, cur_layer_mask], keep_generator=keep_generator)
                layers += [mask_conv_bn, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def get_module_cfg(module_mask, model_cfg):
    # Note: the first conv layer is a normal conv layer rather than a MaskConvBN layer.
    module_cfg = [model_cfg[0]]
    module_layer_masks = []
    point = 0

    for item in model_cfg[1:]:
        if not isinstance(item, int):
            module_cfg.append(item)
        else:
            each_layer_mask = module_mask[point: point + item]
            module_layer_masks.append(each_layer_mask)
            module_cfg.append(torch.sum(each_layer_mask).cpu().int().item())
            point += item
    return module_cfg, module_layer_masks


def get_module_param(module, module_layer_masks, model_param, keep_generator=True):
    """
    get the module's parameters by removing the irrelevant kernels.
    """
    module_param = copy.deepcopy(model_param)

    masked_conv_idx = 0

    for each_layer_idx, each_layer in module.features._modules.items():
        if not isinstance(each_layer, MaskConvBN):
            continue

        layer_mask = module_layer_masks[masked_conv_idx]
        retrained_kernel_indices = torch.nonzero(layer_mask, as_tuple=True)[0]

        for layer_param_name in ['conv.weight', 'conv.bias', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var']:
            full_name = f'features.{each_layer_idx}.{layer_param_name}'
            model_layer_param = model_param[full_name]
            module_layer_param = model_layer_param[retrained_kernel_indices]
            if layer_param_name == 'conv.weight':
                if masked_conv_idx > 0:  # since the first conv layer in the module or model is a normal conv layer.
                    previous_layer_mask = module_layer_masks[masked_conv_idx - 1]
                    previous_retrained_kernel_indices = torch.nonzero(previous_layer_mask, as_tuple=True)[0]
                    module_layer_param = module_layer_param[:, previous_retrained_kernel_indices, :, :]
            module_param[full_name] = module_layer_param

        masked_conv_idx += 1
    assert masked_conv_idx == len(module_layer_masks)

    # modify the first Linear layer's input dimension
    previous_retrained_kernel_indices = torch.nonzero(module_layer_masks[-1], as_tuple=True)[0]
    layer_param_name = 'classifier.0.weight'
    model_linear_weight = model_param[layer_param_name]
    module_param[layer_param_name] = model_linear_weight[:, previous_retrained_kernel_indices]

    if not keep_generator:
        new_module_param = {}
        for param_name in module_param:
            if 'mask_generator' not in param_name:
                new_module_param[param_name] = module_param[param_name]
        module_param = new_module_param
    return module_param


# modified for module reuse.
def _vgg(arch: str, cfg: str, batch_norm: bool,
         model_urls: Dict[str, str],
         module_mask, model_param, keep_generator=True,
         pretrained: bool = False, progress: bool = True,
         **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False

    # Use the module cfg to initialize the module's architecture.
    module_cfg, module_layer_masks = get_module_cfg(module_mask, cfgs[cfg])
    module = VGG(make_layers(module_cfg, batch_norm=batch_norm, module_layer_masks=module_layer_masks, keep_generator=keep_generator),
                 n_last_conv_kernels=module_cfg[-2], **kwargs)

    # initialize the module's parameters with the pretrained model.
    module_param = get_module_param(module, module_layer_masks, model_param, keep_generator=keep_generator)
    module.load_state_dict(module_param)
    return module


def cifar10_vgg16_bn(*args, **kwargs) -> VGG: pass


def cifar100_vgg16_bn(*args, **kwargs) -> VGG: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for cfg, model_name in zip(["A", "B", "D", "E"], ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_vgg,
                    arch=model_name,
                    cfg=cfg,
                    batch_norm=True,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )
