import torch
import torch.nn as nn
from models_cnnsplitter.utils_v2 import MaskConvBN


class SimCNN(nn.Module):
    def __init__(self, num_classes=10, conv_configs=None):
        super(SimCNN, self).__init__()
        self.num_classes = num_classes
        if conv_configs is None:
            conv_configs = [(3, 64), (64, 64),
                            (64, 128), (128, 128),
                            (128, 256), (256, 256), (256, 256),
                            (256, 512), (512, 512), (512, 512),
                            (512, 512), (512, 512), (512, 512)]
        self.conv_configs = conv_configs

        # the name of conv layer must be 'conv_*'
        for i, each_conv_config in enumerate(conv_configs):
            in_channel, out_channel = each_conv_config
            if in_channel == 3:
                setattr(self, f'conv_{i}', nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            else:
                setattr(self, f'conv_{i}', MaskConvBN(in_channel, out_channel, kernel_size=3, padding=1))

        self.dropout_13 = nn.Dropout()
        self.fc_13 = nn.Linear(conv_configs[-1][-1], 512)
        self.dropout_14 = nn.Dropout()
        self.fc_14 = nn.Linear(512, 512)
        self.fc_15 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

        return pred

    def get_masks(self):
        masks = []
        for each_module in self.modules():
            if isinstance(each_module, MaskConvBN):
                masks.append(each_module.masks)
        return masks
