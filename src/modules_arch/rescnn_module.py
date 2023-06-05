import torch
import torch.nn as nn
from models_cnnsplitter.utils_v2 import MaskConvBN
DEVICE = torch.device('cuda')


class ResCNN(nn.Module):
    def __init__(self, module_mask, model_param, keep_generator=True, num_classes=10, conv_configs=None):
        super().__init__()
        self.num_classes = num_classes
        if conv_configs is None:
            conv_configs = [(3, 64), (64, 128),
                            (128, 128), (128, 128),
                            (128, 256), (256, 512),
                            (512, 512), (512, 512),
                            (512, 512), (512, 512),
                            (512, 512), (512, 512)]

        self.residual_idx = [3, 7, 11]
        conv_configs, layer_masks, fc_input_dim = self.get_module_conv_configs(module_mask=module_mask,
                                                                               model_conv_configs=conv_configs)
        self.conv_configs = conv_configs
        self.residual_layer_masks = [(torch.nonzero(layer_masks[i-2], as_tuple=True)[0],
                                      torch.nonzero(layer_masks[i], as_tuple=True)[0]) for i in self.residual_idx]
        self.residual_layer_dim = [(len(layer_masks[i-2]), len(layer_masks[i])) for i in self.residual_idx]
        self.residual_output_mask = [
            torch.nonzero(((layer_masks[i-2] + layer_masks[i]) > 0).int(), as_tuple=True)[0]
            for i in self.residual_idx
        ]

        for i, each_conv_config in enumerate(conv_configs):
            in_channel, out_channel = each_conv_config
            if in_channel == 3:
                setattr(self, f'conv_{i}', nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ))
            else:
                setattr(self, f'conv_{i}', nn.Sequential(
                    MaskConvBN(in_channel, out_channel, kernel_size=3, padding=1, keep_generator=keep_generator),
                    nn.ReLU(inplace=True)
                ))

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.5),
                                        nn.Linear(fc_input_dim, num_classes))

        module_param = self.get_module_param(layer_masks, model_param=model_param)
        self.load_state_dict(module_param)
        self.module_head = None  # set the module head after instantiating the module and loading model params.

    def forward(self, x):
        out = self.conv_0(x)
        out = self.conv_1(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)

        res = out
        out = self.conv_2(out)
        out = self.conv_3(out)
        res = self.padding_for_shortcut(res, self.residual_layer_dim[0][0], self.residual_layer_masks[0][0])
        out = self.padding_for_shortcut(out, self.residual_layer_dim[0][1], self.residual_layer_masks[0][1])
        out += res
        out = out[:, self.residual_output_mask[0], :, :]

        out = self.conv_4(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)
        out = self.conv_5(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)

        res = out
        out = self.conv_6(out)
        out = self.conv_7(out)
        res = self.padding_for_shortcut(res, self.residual_layer_dim[1][0], self.residual_layer_masks[1][0])
        out = self.padding_for_shortcut(out, self.residual_layer_dim[1][1], self.residual_layer_masks[1][1])
        out += res
        out = out[:, self.residual_output_mask[1], :, :]


        out = self.conv_8(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)
        out = self.conv_9(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)

        res = out
        out = self.conv_10(out)
        out = self.conv_11(out)
        res = self.padding_for_shortcut(res, self.residual_layer_dim[2][0], self.residual_layer_masks[2][0])
        out = self.padding_for_shortcut(out, self.residual_layer_dim[2][1], self.residual_layer_masks[2][1])
        out += res
        out = out[:, self.residual_output_mask[2], :, :]

        out = self.classifier(out)
        out = self.module_head(out)
        return out

    def padding_for_shortcut(self, x, padding_dim, layer_mask):
        x_padding = torch.zeros(
            (x.shape[0], padding_dim, x.shape[2], x.shape[3]),
            dtype=x.dtype, device=DEVICE
        )
        x_padding[:, layer_mask, :, :] = x
        return x_padding

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

        # modify the input dimension of the layer following the residual layer.
        for idx in self.residual_idx:
            residual_output_mask = ((layer_masks[idx - 2] + layer_masks[idx]) > 0).int()
            residual_output_dim = torch.sum(residual_output_mask).cpu().item()
            if idx == 11:  # because there is no conv layer following conv_11.
                fc_input_dim = residual_output_dim
                break
            else:
                tmp = module_conv_configs[idx + 1]
                module_conv_configs[idx+1] = (residual_output_dim, tmp[1])

        return module_conv_configs, layer_masks, fc_input_dim

    def get_module_param(self, layer_masks, model_param):
        module_param = dict()

        for param_name in model_param:
            if 'mask' not in param_name:
                module_param[param_name] = model_param[param_name]

        pre_conv_mask = torch.ones(3, dtype=torch.int64)
        pre_retained_kernel_indices = torch.nonzero(pre_conv_mask, as_tuple=True)[0]
        for i, conv_mask in enumerate(layer_masks):
            cur_retained_kernel_indices = torch.nonzero(conv_mask, as_tuple=True)[0]
            if i == 0:
                pre_retained_kernel_indices = cur_retained_kernel_indices
                continue

            for j, residual_idx in enumerate(self.residual_idx):
                if i == residual_idx + 1:  # if the current layer follows the residual conv.
                    pre_retained_kernel_indices = self.residual_output_mask[j]
                    break

            conv_weight = model_param[f'conv_{i}.0.conv.weight']
            module_conv_weight = conv_weight[cur_retained_kernel_indices, :, :, :]
            module_conv_weight = module_conv_weight[:, pre_retained_kernel_indices, :, :]
            module_param[f'conv_{i}.0.conv.weight'] = module_conv_weight

            conv_bias = model_param[f'conv_{i}.0.conv.bias']
            module_conv_bias = conv_bias[cur_retained_kernel_indices]
            module_param[f'conv_{i}.0.conv.bias'] = module_conv_bias

            for bn_name in ('bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var'):
                bn_param = model_param[f'conv_{i}.0.{bn_name}']
                module_bn_param = bn_param[cur_retained_kernel_indices]
                module_param[f'conv_{i}.0.{bn_name}'] = module_bn_param

            pre_retained_kernel_indices = cur_retained_kernel_indices

        pre_retained_kernel_indices = self.residual_output_mask[-1]
        fc_weight = model_param['classifier.2.weight']
        module_param['classifier.2.weight'] = fc_weight[:, pre_retained_kernel_indices]

        return module_param
