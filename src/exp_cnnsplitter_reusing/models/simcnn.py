import math
import torch
import torch.nn as nn


class SimCNN(nn.Module):
    def __init__(self, num_classes=10, conv_configs=None):
        super(SimCNN, self).__init__()
        self.num_classes = num_classes
        is_modular = True
        if conv_configs is None:
            conv_configs = [(3, 64), (64, 64),
                            (64, 128), (128, 128),
                            (128, 256), (256, 256), (256, 256),
                            (256, 512), (512, 512), (512, 512),
                            (512, 512), (512, 512), (512, 512)]
            is_modular = False

        # the name of conv layer must be 'conv_*'
        for i, each_conv_config in enumerate(conv_configs):
            in_channel, out_channel = each_conv_config
            setattr(self, f'conv_{i}', nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))

        self.dropout_13 = nn.Dropout()
        self.fc_13 = nn.Linear(conv_configs[-1][-1], 512)  # for resize 32
        # self.fc_13 = nn.Linear(16 * conv_configs[-1][-1], 512)  # for resize 128
        self.dropout_14 = nn.Dropout()
        self.fc_14 = nn.Linear(512, 512)
        self.fc_15 = nn.Linear(512, num_classes)

        if not is_modular:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

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

    def extract_conv_outputs(self, x):
        conv_outputs = []

        y = torch.relu(self.conv_0(x))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.relu(self.conv_1(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_2(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.relu(self.conv_3(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_4(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.relu(self.conv_5(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.relu(self.conv_6(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_7(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.relu(self.conv_8(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.relu(self.conv_9(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_10(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.relu(self.conv_11(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.relu(self.conv_12(y))
        conv_outputs.append(y.squeeze().cpu().numpy())
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = y.view(y.size(0), -1)
        y = self.dropout_13(y)
        y = torch.relu(self.fc_13(y))

        y = self.dropout_14(y)
        y = torch.relu(self.fc_14(y))

        pred = self.fc_15(y)

        pred_confidence = torch.softmax(pred.squeeze(), 0)
        return conv_outputs, pred_confidence.cpu().numpy()