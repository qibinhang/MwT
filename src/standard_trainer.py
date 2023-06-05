import argparse
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from configs import Configs
from dataset_loader import load_cifar10, load_svhn
from models.resnet import ResNet18
from models.vgg import cifar10_vgg16_bn as vgg16


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['vgg16', 'resnet18'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'svhn'], required=True)

    parser.add_argument('--lr_model', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--hook_conv_outputs', action='store_true')

    args = parser.parse_args()
    return args


def training(model, train_loader, test_loader, num_epoch=200):
    optim = torch.optim.SGD(
        params=model.parameters(), lr=lr_model, momentum=0.9, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epoch)

    for epoch in range(num_epoch):
        print(f'Epoch {epoch}')
        print('-' * 50)

        for data_loader, phase in zip([train_loader, test_loader], ['train', 'test']):
            if not is_hook_conv_outputs:
                with torch.set_grad_enabled(phase == 'train'):
                    acc, loss = _training(model, data_loader, optim, phase)
            else:
                if phase == 'train':
                    with torch.set_grad_enabled(phase == 'train'):
                        acc, loss = _training(model, data_loader, optim, phase)
                else:
                    acc, loss, cohesion, coupling, krr = _test_with_modular_metrics(model, data_loader)
                    writer.add_scalar(f'{phase}/Cohesion', cohesion, epoch)
                    writer.add_scalar(f'{phase}/Coupling', coupling, epoch)
                    writer.add_scalar(f'{phase}/Kernel-Rate', coupling, epoch)

            writer.add_scalar(f'{phase}/Accuracy', acc, epoch)
            writer.add_scalar(f'{phase}/Loss', loss, epoch)

        scheduler.step()
    return model


def _training(model, data_loader, optim, phase):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    log_loss = []
    n_correct, total_labels = 0, 0

    for inputs, labels in tqdm(data_loader, ncols=80):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        pred = torch.argmax(outputs, dim=1)
        n_correct += torch.sum((pred == labels).float())
        total_labels += len(labels)

        if phase == 'train':
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_loss.append(loss.detach())

    return n_correct/total_labels, mean_list(log_loss)


@torch.no_grad()
def _test_with_modular_metrics(model, test_dataset):
    global conv_layer_outputs
    model.eval()

    n_correct, total_labels = 0, 0
    loss_log = []
    cohesion_log, coupling_log, krr_log = [], [], []

    for inputs, labels in tqdm(test_dataset, ncols=80):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss_log.append(loss)

        # calculate cohesion and coupling
        cat_conv_layer_outputs = torch.cat(conv_layer_outputs, dim=1)
        conv_layer_masks = (cat_conv_layer_outputs > 0).float()
        cohesion, coupling, krr = cal_modular_metrics(conv_layer_masks, labels)
        cohesion_log.append(cohesion)
        coupling_log.append(coupling)
        krr_log.append(krr)

        conv_layer_outputs = []

        pred = torch.argmax(outputs, dim=1)
        n_correct += torch.sum((pred == labels).float())
        total_labels += len(labels)

    return n_correct/total_labels, mean_list(loss_log), mean_list(cohesion_log), mean_list(coupling_log), mean_list(krr_log)


def cal_modular_metrics(sample_masks, labels):
    tmp = labels.unsqueeze(0) - labels.unsqueeze(1)
    mask_sim_ground_truth = torch.ones_like(tmp, device='cuda')
    mask_sim_ground_truth[tmp != 0] = 0.0
    mask_sim_ground_truth = mask_sim_ground_truth[
        torch.triu(torch.ones_like(mask_sim_ground_truth, device='cuda'), diagonal=1) == 1]

    # intersection
    sample_masks_trans = sample_masks.T
    intersection_sum = torch.mm(sample_masks, sample_masks_trans)
    intersection_sum = intersection_sum[torch.triu(torch.ones_like(intersection_sum, device='cuda'), diagonal=1) == 1]

    # union
    sample_masks_copy_y = sample_masks.unsqueeze(0)
    sample_masks_copy_x = sample_masks.unsqueeze(1)
    union = sample_masks_copy_x + sample_masks_copy_y
    union = (union > 0).int()
    union_sum = torch.sum(union, dim=-1)
    union_sum = union_sum[torch.triu(torch.ones_like(union_sum, device='cuda'), diagonal=1) == 1]

    # Jaccard Index
    cohesion = intersection_sum / union_sum
    cohesion = cohesion[mask_sim_ground_truth == 1].mean()

    coupling = intersection_sum / union_sum
    coupling = coupling[[mask_sim_ground_truth == 0]].mean()

    # kernel retention rate
    krr = torch.mean(sample_masks.float())

    return cohesion, coupling, krr


def mean_list(input_list):
    return sum(input_list) / len(input_list)


conv_layer_outputs = []
def hook_conv_outputs(module, input, output):
    if not output.requires_grad:
        relu_outputs = torch.relu(output)
        avg_outputs = torch.mean(relu_outputs, dim=(2, 3))
        conv_layer_outputs.append(avg_outputs)


def main():
    if dataset_name == 'cifar10':
        train_loader, test_loader = load_cifar10(configs.dataset_dir, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == 'svhn':
        train_loader, test_loader = load_svhn(f'{configs.dataset_dir}/svhn', batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError

    if model_name == 'vgg16':
        model = vgg16(pretrained=False).to('cuda')
    elif model_name == 'resnet18':
        model = ResNet18().to('cuda')
    else:
        raise ValueError

    if is_hook_conv_outputs:
        for name, layer in model.named_modules():
            if hasattr(layer, 'out_channels'):
                layer.register_forward_hook(hook_conv_outputs)

    model = training(model, train_loader, test_loader, num_epoch=num_epochs)
    torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    args = get_args()
    print(args)

    model_name = args.model
    dataset_name = args.dataset
    lr_model = args.lr_model
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    is_hook_conv_outputs = args.hook_conv_outputs

    num_workers = 2
    configs = Configs()

    log_path = f'{configs.tensorboard_dir}/{model_name}_{dataset_name}/standard_model_lr_{lr_model}_bz_{batch_size}'
    writer = SummaryWriter(log_path)

    model_save_path = f'{configs.data_dir}/{model_name}_{dataset_name}/' \
                      f'standard_model_lr_{lr_model}_bz_{batch_size}.pth'

    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    main()
