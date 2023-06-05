import math
import torch
import torch.nn as nn
from models_cnnsplitter.utils_v2 import MaskConvBN


class SimCNN(nn.Module):
    def __init__(self, module_mask, model_param, keep_generator=True, num_classes=10, conv_configs=None):
        super(SimCNN, self).__init__()
        self.num_classes = num_classes
        if conv_configs is None:
            conv_configs = [(3, 64), (64, 64),
                            (64, 128), (128, 128),
                            (128, 256), (256, 256), (256, 256),
                            (256, 512), (512, 512), (512, 512),
                            (512, 512), (512, 512), (512, 512)]
        conv_configs, layer_masks = self.get_module_conv_configs(module_mask=module_mask,
                                                                 model_conv_configs=conv_configs)
        self.conv_configs = conv_configs

        # the name of conv layer must be 'conv_*'
        for i, each_conv_config in enumerate(conv_configs):
            in_channel, out_channel = each_conv_config
            if i == 0:
                setattr(self, f'conv_{i}', nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            else:
                setattr(self, f'conv_{i}', MaskConvBN(in_channel, out_channel, kernel_size=3, padding=1,
                                                      keep_generator=keep_generator))

        self.dropout_13 = nn.Dropout()
        self.fc_13 = nn.Linear(conv_configs[-1][-1], 512)
        self.dropout_14 = nn.Dropout()
        self.fc_14 = nn.Linear(512, 512)
        self.fc_15 = nn.Linear(512, num_classes)

        module_param = self.get_module_param(layer_masks, model_param=model_param)
        self.load_state_dict(module_param)

        self.module_head = None  # set the module head after instantiating the module and loading model params.

    def forward(self, x):
        y = torch.relu(self.conv_0(x))
        y = torch.relu(self.conv_1(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_2(y))
        y = torch.relu(self.conv_3(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_4(y))
        y = torch.relu(self.conv_5(y))
        y = torch.relu(self.conv_6(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_7(y))
        y = torch.relu(self.conv_8(y))
        y = torch.relu(self.conv_9(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_10(y))
        y = torch.relu(self.conv_11(y))
        y = torch.relu(self.conv_12(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = y.view(y.size(0), -1)
        y = self.dropout_13(y)
        y = torch.relu(self.fc_13(y))

        y = self.dropout_14(y)
        y = torch.relu(self.fc_14(y))

        pred = self.fc_15(y)
        pred = self.module_head(pred)

        return pred

    def get_masks(self):
        masks = []
        for each_module in self.modules():
            if isinstance(each_module, MaskConvBN):
                masks.append(each_module.masks)
        return masks

    def get_module_conv_configs(self, module_mask, model_conv_configs):
        module_conv_configs = [model_conv_configs[0]]
        layer_masks = [torch.ones(module_conv_configs[0][1], dtype=torch.int64)]  # first conv layer does not remove kernels.
        point = 0
        in_channel = module_conv_configs[0][1]
        for _, num_kernels in model_conv_configs[1:]:
            each_layer_mask = module_mask[point: point + num_kernels]
            layer_masks.append(each_layer_mask)

            out_channel = torch.sum(each_layer_mask).cpu().item()
            module_conv_configs.append((in_channel, out_channel))
            in_channel = out_channel

            point += num_kernels
        return module_conv_configs, layer_masks

    def get_module_param(self, layer_masks, model_param):
        module_param = dict()

        for param_name in model_param:
            if 'mask' not in param_name:
                module_param[param_name] = model_param[param_name]

        pre_conv_mask = torch.ones(3, dtype=torch.int64)
        pre_retrained_kernel_indices = torch.nonzero(pre_conv_mask, as_tuple=True)[0]
        for i, conv_mask in enumerate(layer_masks):
            cur_retrained_kernel_indices = torch.nonzero(conv_mask, as_tuple=True)[0]
            if i == 0:
                pre_retrained_kernel_indices = cur_retrained_kernel_indices
                continue

            conv_weight = model_param[f'conv_{i}.conv.weight']
            module_conv_weight = conv_weight[cur_retrained_kernel_indices, :, :, :]
            module_conv_weight = module_conv_weight[:, pre_retrained_kernel_indices, :, :]
            module_param[f'conv_{i}.conv.weight'] = module_conv_weight

            conv_bias = model_param[f'conv_{i}.conv.bias']
            module_conv_bias = conv_bias[cur_retrained_kernel_indices]
            module_param[f'conv_{i}.conv.bias'] = module_conv_bias

            for bn_name in ('bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var'):
                bn_param = model_param[f'conv_{i}.{bn_name}']
                module_bn_param = bn_param[cur_retrained_kernel_indices]
                module_param[f'conv_{i}.{bn_name}'] = module_bn_param

            pre_retrained_kernel_indices = cur_retrained_kernel_indices

        fc_weight = model_param['fc_13.weight']
        module_param['fc_13.weight'] = fc_weight[:, pre_retrained_kernel_indices]

        return module_param
