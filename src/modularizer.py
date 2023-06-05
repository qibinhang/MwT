import argparse
import copy
from models.vgg_masked import cifar10_vgg16_bn as mt_vgg16_model
from modules_arch.vgg_module_v2 import cifar10_vgg16_bn as vgg16_module
from models.vgg import cifar10_vgg16_bn as st_vgg16_model

from models.resnet_masked import ResNet18 as mt_ResNet18_model
from modules_arch.resnet_module_v2 import ResNet18 as ResNet18_module
from models.resnet import ResNet18 as st_ResNet18_model

from models_cnnsplitter.simcnn_masked import SimCNN as mt_simcnn_model
from modules_arch.simcnn_module import SimCNN as simcnn_module
from models_cnnsplitter.simcnn import SimCNN as st_simcnn_model

from models_cnnsplitter.rescnn_masked import ResCNN as mt_rescnn_model
from modules_arch.rescnn_module import ResCNN as rescnn_module
from models_cnnsplitter.rescnn import ResCNN as st_rescnn_model

import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from configs import Configs
from dataset_loader import load_cifar10, load_cifar10_target_class, load_svhn, load_svhn_target_class


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['vgg16', 'resnet18', 'simcnn', 'rescnn'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'svhn'], required=True)

    parser.add_argument('--lr_model', type=float, default=0.05)
    # parser.add_argument('--lr_mask', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--target_classes', nargs='+', type=int, required=True)
    parser.add_argument('--threshold', type=float, default=0.9)

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate_masks_for_samples(modular_model, data_loader):
    modular_model.eval()
    samples_masks = []
    total_labels = []
    for inputs, labels in tqdm(data_loader, ncols=80, desc='masks for samples'):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        modular_model(inputs)
        masks = modular_model.get_masks()
        for conv_idx, each_layer_mask in enumerate(masks):
            if len(samples_masks) == len(masks):
                samples_masks[conv_idx].append(each_layer_mask)
            else:
                samples_masks.append([each_layer_mask])

        total_labels.append(labels)
    return samples_masks, total_labels


@torch.no_grad()
def generate_masks_for_modules(samples_masks, samples_labels, num_classes, mt_model, threshold=0.9):
    masks = mt_model.get_masks()
    num_kernels_of_each_layer = [each_layer_mask.shape[1] for each_layer_mask in masks]

    module_masks = []
    for each_class in tqdm(list(range(num_classes)), ncols=80, desc='masks for modules'):
        all_layer_mask = []
        for each_layer_mask in samples_masks:
            elm = torch.concat(each_layer_mask, dim=0)
            bin_elm = (elm > 0).int()
            target_bin_elm = bin_elm[samples_labels == each_class]
            all_layer_mask.append(target_bin_elm)
        all_layer_mask = torch.cat(all_layer_mask, dim=1)
        frequency = torch.sum(all_layer_mask, dim=0) / all_layer_mask.shape[0]

        point = 0
        all_layer_frequency = []
        for n_kernels in num_kernels_of_each_layer:
            all_layer_frequency.append(frequency[point: point + n_kernels])
            point += n_kernels
        assert point == len(frequency)
        each_module_mask = []
        for layer_frequency in all_layer_frequency:
            cur_thres = threshold
            while True:
                layer_mask = (layer_frequency >= cur_thres).int()
                if torch.sum(layer_mask) == 0.0:
                    cur_thres -= 0.05
                    if cur_thres <= 0:
                        raise ValueError(f'cur_thres = {cur_thres} should greater than 0.0')
                else:
                    break
            each_module_mask.append(layer_mask)
        each_module_mask = torch.cat(each_module_mask, dim=0)
        module_masks.append(each_module_mask)
    module_masks = torch.stack(module_masks, dim=0)
    return module_masks


def cal_jaccard_index(masks):
    n_masks = masks.shape[0]
    circle_idx = list(range(n_masks))
    results = []
    for i in range(n_masks-1):
        circle_idx = circle_idx[1:] + [circle_idx[0]]
        circle_mask = masks[circle_idx]
        tmp = masks * circle_mask
        intersection = torch.sum(tmp, dim=1)
        tmp = ((masks + circle_mask) > 0).int()
        union = torch.sum(tmp, dim=1)
        jaccard_index = intersection / union
        results.append(torch.mean(jaccard_index))
    return torch.mean(torch.stack(results))


def eval_module_cohesion(samples_masks, modules_masks, samples_labels, num_classes):
    cohesion = []
    for each_class in tqdm(list(range(num_classes)), ncols=80, desc='eval cohesion'):
        all_layer_mask = []
        for each_layer_mask in samples_masks:
            elm = torch.cat(each_layer_mask, dim=0)
            bin_elm = (elm > 0).int()
            target_bin_elm = bin_elm[samples_labels == each_class]
            all_layer_mask.append(target_bin_elm)
        all_layer_mask = torch.cat(all_layer_mask, dim=1)

        # filter some kernels which are removed by THRESHOLD in generating the module mask.
        module_mask = modules_masks[each_class]
        all_layer_mask = all_layer_mask * module_mask.unsqueeze(0)

        each_class_cohesion = cal_jaccard_index(all_layer_mask)
        cohesion.append(each_class_cohesion)
    return torch.mean(torch.stack(cohesion))


def eval_module_coupling(modules_masks):
    print(f'eval coupling...')
    coupling = cal_jaccard_index(modules_masks)
    return coupling


def eval_module_metric(samples_masks, modules_masks, samples_labels, num_classes):
    cohesion = eval_module_cohesion(samples_masks, modules_masks, samples_labels, num_classes)
    coupling = eval_module_coupling(modules_masks)
    return cohesion, coupling


@torch.no_grad()
def evaluate_model(model, data_loader, target_classes: list =None):
    model.eval()
    n_correct, total_labels = 0, 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        if target_classes is not None:
            outputs = outputs[:, target_classes]

        predicts = torch.argmax(outputs, dim=1)
        n_correct += torch.sum((predicts == labels).float())
        total_labels += len(labels)
    return n_correct/total_labels


def mean_list(input_list):
    return sum(input_list) / len(input_list)


def fine_tune_module(module, train_loader, test_loader, num_epoch=5):
    optim = torch.optim.SGD(
        params=module.parameters(), lr=lr_model, momentum=0.9, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epoch * 2)

    best_acc = 0.0
    best_module = None

    # begin modular training
    for epoch in range(num_epoch):
        print(f'Epoch {epoch}')
        print('-' * 50)

        acc, loss = train_module(module, train_loader, optim)
        print(f'[Train]  ACC: {acc:.2%}  |  Loss: {loss:.3f}')

        acc = evaluate_module(module, test_loader)
        print(f'[Test]  ACC: {acc:.2%}')

        if acc > best_acc:
            best_acc = acc
            best_module = copy.deepcopy(module)
        scheduler.step()
    return best_module


def train_module(model, train_loader, optim):
    model.train()
    log_loss = []
    n_correct, total_labels = 0, 0

    for inputs, labels in tqdm(train_loader, ncols=80):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        pred = torch.argmax(outputs, dim=1)
        n_correct += torch.sum((pred == labels).float())
        total_labels += len(labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        log_loss.append(loss.detach())

    return n_correct/total_labels, mean_list(log_loss)


@torch.no_grad()
def evaluate_module(module, test_loader):
    module.eval()
    n_corrects, n_tc_labels = 0, 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = module(inputs)
        predicts = torch.argmax(outputs, dim=1)
        n_corrects += torch.sum(predicts == labels)
        n_tc_labels += len(labels)
    acc = n_corrects / n_tc_labels
    return acc


def generate_all_modules_masks():
    if model_name == 'vgg16':
        mt_model = mt_vgg16_model(pretrained=False).to(DEVICE)
    elif model_name == 'resnet18':
        mt_model = mt_ResNet18_model(num_classes=10).to(DEVICE)
    elif model_name == 'simcnn':
        mt_model = mt_simcnn_model().to(DEVICE)
    elif model_name == 'rescnn':
        mt_model = mt_rescnn_model().to(DEVICE)
    else:
        raise ValueError

    if dataset_name == 'cifar10':
        train_loader, test_loader = load_cifar10(configs.dataset_dir, batch_size=batch_size, num_workers=2)
    elif dataset_name == 'svhn':
        train_loader, test_loader = load_svhn(f'{configs.dataset_dir}/svhn', batch_size=batch_size, num_workers=2)
    else:
        raise ValueError

    mt_model.load_state_dict(torch.load(mt_model_save_path, map_location=DEVICE))

    # Check the model's ACC
    acc = evaluate_model(mt_model, test_loader)
    print(f'Check the Modular Model ACC: {acc:.2%}\n')

    samples_masks, samples_labels = generate_masks_for_samples(mt_model, train_loader)
    samples_labels = torch.cat(samples_labels, dim=0)
    modules_masks = generate_masks_for_modules(samples_masks, samples_labels, mt_model.num_classes,
                                               mt_model, THRESHOLD)
    torch.save(modules_masks, modules_masks_save_path)

    # evaluate modularization on metrics of kernel retention rate, cohesion, and coupling.
    cohesion, coupling = eval_module_metric(samples_masks, modules_masks, samples_labels, mt_model.num_classes)
    module_kernel_rate = torch.mean(modules_masks.float(), dim=1)
    module_kernel_rate = torch.mean(module_kernel_rate)
    print(f'Module_Kernel_Rate={module_kernel_rate:.4f}  |  Cohesion={cohesion:.4f}  |  Coupling={coupling:.4f}')


def generate_target_module(target_classes, module_mask_path):
    # load modules' masks and the pretrained model
    all_modules_masks = torch.load(module_mask_path, map_location=DEVICE)
    target_module_mask = (torch.sum(all_modules_masks[target_classes], dim=0) > 0).int()
    # generate modules_arch by removing kernels from the model.
    mt_model_param = torch.load(mt_model_save_path, map_location=DEVICE)
    kernel_rate = torch.sum(target_module_mask) / len(target_module_mask)
    print(f'Kernel Rate: {kernel_rate:.2%}')

    if model_name == 'vgg16':
        module = vgg16_module(model_param=mt_model_param, module_mask=target_module_mask,
                              keep_generator=False).to(DEVICE)
    elif model_name == 'resnet18':
        module = ResNet18_module(model_param=mt_model_param, module_mask=target_module_mask,
                                 keep_generator=False).to(DEVICE)
    elif model_name == 'simcnn':
        module = simcnn_module(model_param=mt_model_param, module_mask=target_module_mask,
                               keep_generator=False).to(DEVICE)
    elif model_name == 'rescnn':
        module = rescnn_module(model_param=mt_model_param, module_mask=target_module_mask,
                               keep_generator=False).to(DEVICE)
    else:
        raise ValueError

    module.module_head = torch.nn.Sequential(
        torch.nn.ReLU(True),
        torch.nn.Linear(module.num_classes, len(target_classes)),
    ).to(DEVICE)
    return module


def main():
    if not os.path.exists(modules_masks_save_path):
        generate_all_modules_masks()

    # load the target module
    module = generate_target_module(target_classes, modules_masks_save_path)
    if dataset_name == 'cifar10':
        target_train_loader, target_test_loader = load_cifar10_target_class(
            configs.dataset_dir, batch_size=batch_size, num_workers=num_workers, target_classes=target_classes)
    elif dataset_name == 'svhn':
        target_train_loader, target_test_loader = load_svhn_target_class(
            f'{configs.dataset_dir}/svhn', batch_size=batch_size, num_workers=num_workers, target_classes=target_classes)
    else:
        raise ValueError

    # fine-tune the target module
    module = fine_tune_module(module, target_train_loader, target_test_loader, num_epoch=10)
    fine_tuned_acc = evaluate_module(module, target_test_loader)
    print(f'Module ACC (fine-tuned): {fine_tuned_acc:.2%}\n')
    torch.save(module.state_dict(), modules_save_path)

    # compared to the standard model
    if model_name == 'vgg16':
        st_model = st_vgg16_model(pretrained=False).to(DEVICE)
    elif model_name == 'resnet18':
        st_model = st_ResNet18_model().to(DEVICE)
    elif model_name == 'simcnn':
        st_model = st_simcnn_model().to(DEVICE)
    elif model_name == 'rescnn':
        st_model = st_rescnn_model().to(DEVICE)
    else:
        raise ValueError
    st_model.load_state_dict(torch.load(st_model_save_path, map_location=DEVICE))
    st_model_acc = evaluate_model(st_model, target_test_loader, target_classes=target_classes)
    print(f'Standard Model ACC     : {st_model_acc:.2%}\n')


if __name__ == '__main__':
    args = get_args()
    DEVICE = torch.device('cuda')

    print(args)
    print('-'*100)
    model_name = args.model
    dataset_name = args.dataset
    lr_model = args.lr_model
    lr_mask = lr_model
    # lr_mask = args.lr_mask
    alpha = args.alpha
    beta = args.beta
    batch_size = args.batch_size
    THRESHOLD = args.threshold
    target_classes = args.target_classes
    print(f'TCs: {target_classes}')

    num_workers = 2

    configs = Configs()

    log_path = f'{configs.tensorboard_dir}/lr_model_mask_{lr_model}_{lr_mask}_a_{alpha}_b_{beta}_bz_{batch_size}'
    save_dir = f'{configs.data_dir}/{model_name}_{dataset_name}'
    mt_model_save_path = f'{save_dir}/lr_model_mask_{lr_model}_{lr_mask}_a_{alpha}_b_{beta}_bz_{batch_size}.pth'
    modules_masks_save_path = f'{save_dir}/modules/' \
                              f'lr_model_mask_{lr_model}_{lr_mask}_a_{alpha}_b_{beta}_bz_{batch_size}/mask_thres_{THRESHOLD}.pth'

    tc_str = ''.join([str(tc) for tc in target_classes])
    modules_save_path = f'{save_dir}/modules/' \
                        f'lr_model_mask_{lr_model}_{lr_mask}_a_{alpha}_b_{beta}_bz_{batch_size}/module_tc_{tc_str}_thres_{THRESHOLD}.pth'

    mt_model_save_dir = os.path.dirname(mt_model_save_path)
    if not os.path.exists(mt_model_save_dir):
        os.makedirs(mt_model_save_dir)

    modules_masks_save_dir = os.path.dirname(modules_masks_save_path)
    if not os.path.exists(modules_masks_save_dir):
        os.makedirs(modules_masks_save_dir)

    st_model_save_path = f'{save_dir}/standard_model_lr_0.05_bz_128.pth'
    assert os.path.exists(st_model_save_path)

    main()
