# from https://github.com/ZOMIN28/ResNet18_Cifar10_95.46/blob/main/utils/ResNet.py
# Thanks the author.

import torch
import torch.nn as nn
from models.utils_v2 import MaskConvBN
DEVICE = torch.device('cuda')


def mask_conv3x3_bn(in_planes, out_planes, stride=1, groups=1, dilation=1,
                    module_layer_masks=None, keep_generator=True):
    """3x3 convolution with padding"""
    return MaskConvBN(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=False, dilation=dilation,
                      module_layer_masks=module_layer_masks, keep_generator=keep_generator)


def mask_conv1x1_bn(in_planes, out_planes, stride=1, module_layer_masks=None, keep_generator=True):
    """1x1 convolution"""
    return MaskConvBN(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False,
                      module_layer_masks=module_layer_masks, keep_generator=keep_generator)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, block_masks=None, keep_generator=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        conv1_inplanes = int(sum(block_masks['pre_block_last_conv_mask']))
        conv1_planes = int(sum(block_masks['conv1']))
        self.conv1 = mask_conv3x3_bn(conv1_inplanes, conv1_planes, stride,
                                     module_layer_masks=[block_masks['pre_block_last_conv_mask'], block_masks['conv1']],
                                     keep_generator=keep_generator)
        self.relu = nn.ReLU(inplace=True)

        conv2_planes = int(sum(block_masks['conv2']))
        self.conv2 = mask_conv3x3_bn(conv1_planes, conv2_planes,
                                     module_layer_masks=[block_masks['conv1'], block_masks['conv2']],
                                     keep_generator=keep_generator)

        # self.conv1 = mask_conv3x3_bn(inplanes, planes, stride)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = mask_conv3x3_bn(planes, planes)

        self.downsample = downsample
        self.stride = stride

        identity_mask = block_masks['downsample'] if 'downsample' in block_masks else block_masks['pre_block_last_conv_mask']
        self.identity_padding_dim = len(identity_mask)
        self.identity_mask = torch.nonzero(identity_mask, as_tuple=True)[0]
        self.conv2_mask = torch.nonzero(block_masks['conv2'], as_tuple=True)[0]
        self.residual_output_mask = torch.nonzero(block_masks['cur_residual_output_mask'], as_tuple=True)[0]

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # padding identify and out.
        identity_padding = torch.zeros(
            (identity.shape[0], self.identity_padding_dim, identity.shape[2], identity.shape[3]),
            dtype=identity.dtype, device=DEVICE
        )
        identity_padding[:, self.identity_mask, :, :] = identity

        out_padding = torch.zeros(
            (out.shape[0], self.identity_padding_dim, out.shape[2], out.shape[3]),
            dtype=out.dtype, device=DEVICE
        )
        out_padding[:, self.conv2_mask, :, :] = out

        out_padding += identity_padding
        out = out_padding[:, self.residual_output_mask, :, :]

        # out += identity

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    #     self.conv1 = conv1x1(inplanes, width)
    #     self.bn1 = norm_layer(width)
    #     self.conv2 = conv3x3(width, width, stride, groups, dilation)
    #     self.bn2 = norm_layer(width)
    #     self.conv3 = conv1x1(width, planes * self.expansion)
    #     self.bn3 = norm_layer(planes * self.expansion)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.downsample = downsample
    #     self.stride = stride
    #
    # def forward(self, x):
    #     identity = x
    #
    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #
    #     out = self.conv2(out)
    #     out = self.bn2(out)
    #     out = self.relu(out)
    #
    #     out = self.conv3(out)
    #     out = self.bn3(out)
    #
    #     if self.downsample is not None:
    #         identity = self.downsample(x)
    #
    #     out += identity
    #     out = self.relu(out)
    #
    #     return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, detailed_masks=None, keep_generator=True):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # For CIFAR-10 with resolution of 32 x 32
        """
        ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息,
        所以在实验中我们将7x7的降采样层和最大池化层去掉,替换为一个3x3的降采样卷积,
        同时减小该卷积层的步长和填充大小
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       layer_masks=detailed_masks['layer1'],
                                       keep_generator=keep_generator)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       layer_masks=detailed_masks['layer2'],
                                       keep_generator=keep_generator)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       layer_masks=detailed_masks['layer3'],
                                       keep_generator=keep_generator)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       layer_masks=detailed_masks['layer4'],
                                       keep_generator=keep_generator)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = torch.nn.Linear(int(sum(detailed_masks['layer_fc']['pre_block_last_conv_mask'])), num_classes)

        self.module_head = None  # set the module head after instantiating the module and loading model params.

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = torch.nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    layer_masks=None, keep_generator=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            in_channels = int(sum(layer_masks['0']['pre_block_last_conv_mask']))
            out_channels = int(sum(layer_masks['0']['downsample']))
            downsample = nn.Sequential(
                mask_conv1x1_bn(in_channels, out_channels, stride,
                                module_layer_masks=[layer_masks['0']['pre_block_last_conv_mask'], layer_masks['0']['downsample']],
                                keep_generator=keep_generator),
                # mask_conv1x1_bn(self.inplanes, planes * block.expansion, stride),
                # norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(None, None, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, block_masks=layer_masks['0'],
                            keep_generator=keep_generator))

        for block_idx in range(1, blocks):
            layers.append(block(None, None, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, block_masks=layer_masks[str(block_idx)],
                                keep_generator=keep_generator))

        # self.inplanes = planes * block.expansion
        # for _ in range(1, blocks):
        #     layers.append(block(self.inplanes, planes, groups=self.groups,
        #                         base_width=self.base_width, dilation=self.dilation,
        #                         norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.module_head(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def get_masks(self):
        masks = []
        for each_module in self.modules():
            if isinstance(each_module, MaskConvBN):
                masks.append(each_module.masks)
        return masks


def _resnet(block, layers, detailed_masks, **kwargs):
    module = ResNet(block, layers, detailed_masks=detailed_masks, **kwargs)
    return module


# modify for the ResNet18 module.
def ResNet18(model_param, module_mask, keep_generator=True, **kwargs):
    from models.resnet_masked import ResNet18 as mt_ResNet18_Model
    mt_model = mt_ResNet18_Model()
    detailed_masks = get_detailed_masks(mt_model, module_mask)

    module = _resnet(BasicBlock, [2, 2, 2, 2], detailed_masks=detailed_masks, keep_generator=keep_generator, **kwargs)
    module_param = get_module_param(model_param, detailed_masks, keep_generator=keep_generator)
    module.load_state_dict(module_param)
    return module


def ResNet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3],**kwargs)


def ResNet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3],**kwargs)


def ResNet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3],**kwargs)


def get_detailed_masks(model, module_mask):
    """
    {
    'layer1': {'block1': {'conv1': , 'conv2': , 'pre_block_last_conv_mask'},
               'block2': ,...
                },
    'layer2': {'block1': {'conv1': , 'conv2': , 'downsample': , 'pre_block_last_conv_mask'}},
    'layer3': {},
    'layer4': {},
    }
    """

    detailed_masks = {}
    mask_point = 0
    for name, each_module in model.named_modules():
        if isinstance(each_module, MaskConvBN):
            detail_name = name.split('.')
            layer_name, block_name, conv_name = detail_name[0], detail_name[1], detail_name[2]
            layer_masks = detailed_masks.get(layer_name, {})
            block_masks = layer_masks.get(block_name, {})
            num_kernels = each_module.conv.out_channels

            block_masks[conv_name] = module_mask[mask_point: mask_point + num_kernels]
            layer_masks[block_name] = block_masks
            detailed_masks[layer_name] = layer_masks

            mask_point += num_kernels

    # NOTE: the codes below are only applicable to BasicBlock.
    pre_block_last_conv_mask = torch.ones(64, device=DEVICE)
    for layer_idx in range(1, 5):
        for block_idx in range(2):
            detailed_masks[f'layer{layer_idx}'][str(block_idx)]['pre_block_last_conv_mask'] = pre_block_last_conv_mask

            # For residual connection
            conv_mask = detailed_masks[f'layer{layer_idx}'][str(block_idx)]['conv2']
            if 'downsample' in detailed_masks[f'layer{layer_idx}'][str(block_idx)]:
                downsample_mask = detailed_masks[f'layer{layer_idx}'][str(block_idx)]['downsample']
                cur_residual_output_mask = ((conv_mask + downsample_mask) > 0).int()
            else:
                cur_residual_output_mask = ((conv_mask + pre_block_last_conv_mask) > 0).int()

            detailed_masks[f'layer{layer_idx}'][str(block_idx)]['cur_residual_output_mask'] = cur_residual_output_mask
            pre_block_last_conv_mask = cur_residual_output_mask

    detailed_masks['layer_fc'] = {'pre_block_last_conv_mask': pre_block_last_conv_mask}

    return detailed_masks


def get_module_param(model_param, detailed_masks, keep_generator=True):
    module_param = {}
    for each_param_name in model_param:
        if 'mask_generator' in each_param_name or 'num_batches_tracked' in each_param_name:
            continue
        detailed_name = each_param_name.split('.')
        try:
            layer_name, block_name, conv_name = detailed_name[0], detailed_name[1], detailed_name[2]
            conv_bn_mask = detailed_masks[layer_name][block_name][conv_name]
            conv_bn_mask = torch.nonzero(conv_bn_mask, as_tuple=True)[0]
            if conv_name in ('conv1', 'downsample'):
                pre_conv_bn_mask = detailed_masks[layer_name][block_name]['pre_block_last_conv_mask']
            elif conv_name == 'conv2':
                pre_conv_bn_mask = detailed_masks[layer_name][block_name]['conv1']
            else:
                raise ValueError('Only supporting BasicBlock')
            pre_conv_bn_mask = torch.nonzero(pre_conv_bn_mask, as_tuple=True)[0]
        except:
            continue

        # Remove redundant conv kernels and bn weights
        init_param = model_param[each_param_name]
        conv_or_bn = detailed_name[4] if detailed_name[2] == 'downsample' else detailed_name[3]
        if conv_or_bn == 'conv':
            masked_param = init_param[conv_bn_mask]
            masked_param = masked_param[:, pre_conv_bn_mask, :, :]
        elif conv_or_bn == 'bn':
            masked_param = init_param[conv_bn_mask]
        else:
            raise ValueError("conv_or_bn is expected to be 'conv' or 'bn'")
        module_param[each_param_name] = masked_param

    fc_pre_block_last_conv_mask = detailed_masks['layer_fc']['pre_block_last_conv_mask']
    fc_pre_block_last_conv_mask = torch.nonzero(fc_pre_block_last_conv_mask, as_tuple=True)[0]
    fc_weight = model_param['fc.weight']
    module_param['fc.weight'] = fc_weight[:, fc_pre_block_last_conv_mask]
    model_param.update(module_param)
    module_param = model_param

    if not keep_generator:
        new_module_param = {}
        for param_name in module_param:
            if 'mask_generator' not in param_name:
                new_module_param[param_name] = module_param[param_name]
        module_param = new_module_param

    return module_param
