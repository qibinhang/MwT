import argparse
import torch
from module_tools import load_modules
from configure_loader import load_configure
from dataset_loader import load_cifar10_target_class, load_svhn_target_class
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_modules(tc_modules, tc_dataset, tcs):
    n_corrects, n_tc_labels = 0, 0
    for inputs, labels in tc_dataset:
        inputs, labels = inputs.to(device), labels.to(device)
        final_output = []
        for i, each_module in enumerate(tc_modules):
            tc = tcs[i]
            m_output = each_module(inputs)
            m_tc_output = m_output[:, tc]
            final_output.append(m_tc_output.unsqueeze(-1))
        final_output = torch.cat(final_output, dim=1)
        final_pred = torch.argmax(final_output, dim=1)
        n_corrects += torch.sum((final_pred == labels).int())
        n_tc_labels += len(labels)
    acc = n_corrects / n_tc_labels
    return acc


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', choices=['simcnn', 'rescnn'])
    parse.add_argument('--dataset', choices=['cifar10', 'svhn'])
    parse.add_argument('--target_classes', nargs='+', type=int, required=True)
    args = parse.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    target_classes = args.target_classes

    configs = load_configure(model_name, dataset_name)
    modules = load_modules(configs)

    target_modules = [modules[tc][0].eval() for tc in target_classes]

    dataset_dir = 'YourDatasetDir'
    if dataset_name == 'cifar10':
        target_train_loader, target_test_loader = load_cifar10_target_class(
            dataset_dir, batch_size=128, num_workers=0, target_classes=target_classes)
    elif dataset_name == 'svhn':
        target_train_loader, target_test_loader = load_svhn_target_class(
            f'{dataset_dir}/svhn', batch_size=128, num_workers=0, target_classes=target_classes)
    else:
        raise ValueError

    acc = evaluate_modules(target_modules, target_test_loader, target_classes)
    print(f'ACC: {acc:.2%}')

    print()
    n_kernels_model = 4224 if model_name == 'simcnn' else 4228
    target_modules_kernels = [modules[tc][1] for tc in target_classes]
    n_kernels_modules = sum([len(each_tmk) for each_tmk in target_modules_kernels])
    print(f'KRR: {n_kernels_modules/n_kernels_model:.2%}  n_kernels: {n_kernels_modules}')