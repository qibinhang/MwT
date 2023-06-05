import argparse
import torch
from module_tools import load_modules
from configure_loader import load_configure
from dataset_loader import load_cifar10_target_class, load_svhn_target_class
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import sys
import math
sys.path.append('..')

conv_layer_outputs = []
def hook_conv_outputs(module, input, output):
    if not output.requires_grad:
        relu_outputs = torch.relu(output)
        avg_outputs = torch.mean(relu_outputs, dim=(2, 3))
        conv_layer_outputs.append(avg_outputs)


def calculate_cohesion(sample_masks):
    # intersection
    sample_masks_trans = sample_masks.T
    intersection_sum = torch.mm(sample_masks, sample_masks_trans)
    intersection_sum = intersection_sum[torch.triu(torch.ones_like(intersection_sum, device='cuda'), diagonal=1) == 1]

    # union
    union_list = []
    for i in range(sample_masks.shape[0]):
        each_union_sum = sample_masks[i] + sample_masks
        each_union_sum = (each_union_sum > 0).int()
        each_union_sum = torch.sum(each_union_sum, dim=-1)
        union_list.append(each_union_sum.unsqueeze(0))
    union_sum = torch.cat(union_list, dim=0)
    union_sum = union_sum[torch.triu(torch.ones_like(union_sum, device='cuda'), diagonal=1) == 1]

    # Jaccard Index
    cohesion = intersection_sum / union_sum
    return cohesion.mean()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', choices=['simcnn', 'rescnn'])
    parse.add_argument('--dataset', choices=['cifar10', 'svhn'])
    args = parse.parse_args()
    model_name = args.model
    dataset_name = args.dataset

    configs = load_configure(model_name, dataset_name)
    modules = load_modules(configs)

    cohesion_log = []

    for TC in range(10):
        target_module = modules[TC][0].eval()
        for name, layer in target_module.named_modules():
            if hasattr(layer, 'out_channels'):
                layer.register_forward_hook(hook_conv_outputs)

        dataset_dir = 'YourDatasetDir'
        if dataset_name == 'cifar10':
            target_train_loader, target_test_loader = load_cifar10_target_class(
                dataset_dir, batch_size=128, num_workers=0, target_classes=[TC])
        elif dataset_name == 'svhn':
            target_train_loader, target_test_loader = load_svhn_target_class(
                f'{dataset_dir}/svhn', batch_size=128, num_workers=0, target_classes=[TC])
        else:
            raise ValueError

        with torch.no_grad():
            all_batch_conv_layer_outputs = []
            for inputs, labels in target_test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = target_module(inputs)
                cat_conv_layer_outputs = torch.cat(conv_layer_outputs, dim=1)
                all_batch_conv_layer_outputs.append(cat_conv_layer_outputs)
                conv_layer_outputs = []

        cat_all_batch_conv_layer_outputs = torch.cat(all_batch_conv_layer_outputs, dim=0)
        conv_layer_masks = (cat_all_batch_conv_layer_outputs > 0).to(torch.float16)
        tc_cohesion = calculate_cohesion(conv_layer_masks)
        print(tc_cohesion)
        cohesion_log.append(tc_cohesion)

    print()
    print(sum(cohesion_log) / 10)