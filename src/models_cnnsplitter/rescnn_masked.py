import torch
import torch.nn as nn
from models_cnnsplitter.utils_v2 import MaskConvBN


class ResCNN(nn.Module):
    def __init__(self, num_classes=10, conv_configs=None):
        super().__init__()
        self.num_classes = num_classes
        if conv_configs is None:
            conv_configs = [(3, 64), (64, 128),
                            (128, 128), (128, 128),
                            (128, 256), (256, 512),
                            (512, 512), (512, 512),
                            (512, 512), (512, 512),
                            (512, 512), (512, 512)]
        self.conv_configs = conv_configs
        self.residual_idx = [3, 7, 11]

        for i, each_conv_config in enumerate(conv_configs):
            in_channel, out_channel = each_conv_config
            if in_channel == 3:
                setattr(self, f'conv_{i}', nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ))
            else:
                setattr(self, f'conv_{i}', nn.Sequential(
                    MaskConvBN(in_channel, out_channel, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ))

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.5),
                                        nn.Linear(conv_configs[-1][1], num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_0(x)
        out = self.conv_1(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)

        res = out
        out = self.conv_2(out)
        out = self.conv_3(out) + res

        out = self.conv_4(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)
        out = self.conv_5(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)

        res = out
        out = self.conv_6(out)
        out = self.conv_7(out) + res

        out = self.conv_8(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)
        out = self.conv_9(out)
        out = torch.max_pool2d(out, kernel_size=2, stride=2)

        res = out
        out = self.conv_10(out)
        out = self.conv_11(out) + res

        out = self.classifier(out)
        return out

    def get_masks(self):
        masks = []
        for each_module in self.modules():
            if isinstance(each_module, MaskConvBN):
                masks.append(each_module.masks)
        return masks
