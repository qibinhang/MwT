import numpy as np
import torch
import _pickle as pickle
import re
from tqdm import tqdm
from model_loader import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_module(conv_info, model):
    if model.__class__.__name__ == 'SimCNN':
        module = _extract_module_for_simcnn(conv_info, model)
    elif model.__class__.__name__ == 'ResCNN':
        module = _extract_module_for_rescnn(conv_info, model)
    else:
        raise ValueError

    active_kernel_idx = []
    for conv_idx in conv_info:
        each_conv_active_kernel_idx = conv_info[conv_idx]
        for kernel_idx in each_conv_active_kernel_idx:
            active_kernel_idx.append(f'{conv_idx}_k_{kernel_idx}')
    return module, active_kernel_idx


def _extract_module_for_simcnn(conv_info, model):
    """
    conv_info: {'conv_i': [k_idx, ...],}
    """
    from models.simcnn import SimCNN
    n_conv = len([1 for layer_name in model._modules if 'conv' in layer_name])
    assert len(conv_info) == n_conv
    # get the configures of module from conv_info
    conv_configs = [0] * n_conv
    total_conv_names = [f'conv_{idx}' for idx in range(n_conv)]  # for sorting conv_names in conv_info
    for conv_name in total_conv_names:
        idx = int(conv_name[5:])
        if len(conv_info[conv_name]) == 0:
            conv_info[conv_name] = [0]
            print(f'WARNING: {conv_name} has no active kernels.')
        conv_configs[idx] = len(conv_info[conv_name])

    cin = 3
    for i, cout in enumerate(conv_configs):
        conv_configs[i] = (cin, cout)
        cin = cout

    # initialize module
    module = SimCNN(conv_configs=conv_configs, num_classes=model.num_classes)
    # extract the parameters of active kernels from model
    active_kernel_param = {}
    model_param = model.state_dict()
    for i in range(n_conv):
        conv_weight = model_param[f'conv_{i}.weight']
        conv_bias = model_param[f'conv_{i}.bias']

        cur_conv_active_kernel_idx = list(sorted(conv_info[f'conv_{i}']))  # active Cout
        pre_conv_active_kernel_idx = list(sorted(conv_info[f'conv_{i-1}'])) if i > 0 else list(range(3))  # active Cin

        tmp = conv_weight[cur_conv_active_kernel_idx, :, :, :]
        active_kernel_param[f'conv_{i}.weight'] = tmp[:, pre_conv_active_kernel_idx, :, :]
        active_kernel_param[f'conv_{i}.bias'] = conv_bias[cur_conv_active_kernel_idx]

    first_fc_weight = model_param[f'fc_{n_conv}.weight']
    pre_conv_active_kernel_idx = list(sorted(conv_info[f'conv_{n_conv-1}']))
    first_fc_weight = torch.reshape(first_fc_weight,
                                    (first_fc_weight.size(0),
                                     model_param[f'conv_{n_conv-1}.bias'].size(0), -1))  # (512, 512, 16)
    active_first_fc_weight = first_fc_weight[:, pre_conv_active_kernel_idx, :]
    active_first_fc_weight = torch.reshape(active_first_fc_weight, (active_first_fc_weight.size(0), -1))
    active_kernel_param[f'fc_{n_conv}.weight'] = active_first_fc_weight

    model_param.update(active_kernel_param)
    module.load_state_dict(model_param)
    module = module.to(device).eval()
    return module


def _extract_module_for_rescnn(conv_info, model):
    # init module
    from models.rescnn import ResCNN
    conv_block_configs = []
    cin = 3
    pool = [False, True, False, False, True, True, False, False, True, True, False, False]
    total_conv_names = [f'conv_{idx}' for idx in range(len(conv_info))]
    for idx, conv_block_name in enumerate(total_conv_names):
        if len(conv_info[conv_block_name]) == 0:
            print(f'WARNING: {conv_block_name} has no active kernels.')
            conv_info[conv_block_name] = [0]
        conv_block_configs.append((cin, len(conv_info[conv_block_name]), pool[idx]))
        cin = len(conv_info[conv_block_name])
    module = ResCNN(block_configs=conv_block_configs, num_classes=model.num_classes)
    module.to(device)

    # extract the parameters of active kernels from model
    model_param = model.state_dict()
    active_param = {}
    pre = None
    for conv_name in total_conv_names:
        cur_active_kernel_idx = list(sorted(conv_info[conv_name]))
        pre_active_kernel_idx = list(sorted(conv_info[pre])) if pre is not None else list(range(3))
        conv_weight = model_param[f'{conv_name}.0.weight']
        tmp = conv_weight[cur_active_kernel_idx, :, :, :]
        active_param[f'{conv_name}.0.weight'] = tmp[:, pre_active_kernel_idx, :, :]
        conv_bias = model_param[f'{conv_name}.0.bias']
        active_param[f'{conv_name}.0.bias'] = conv_bias[cur_active_kernel_idx]
        pre = conv_name

    pre_active_kernel_idx = list(sorted(conv_info[total_conv_names[-1]]))
    fc_weight = model_param['classifier.2.weight']
    active_param['classifier.2.weight'] = fc_weight[:, pre_active_kernel_idx]
    model_param.update(active_param)

    module.load_state_dict(model_param)
    return module.eval()


def cal_fitness(solution, model, target_class, dataset, configs):
    conv_info = decode(solution, configs)
    module, active_kernel_idx = extract_module(conv_info, model)
    outputs, labels = module_predict(module, dataset)
    outputs = outputs[:, target_class].unsqueeze(-1)
    return outputs.cpu().numpy(), labels.cpu().numpy(), active_kernel_idx


def decode(solution, configs):
    """
    transfer the solution to the module and the conv_info.
    conv_info is used to extract the module and will be used to calculated diff in module_recorder.py
    """
    # transform the entire model solution to each conv layer solution.
    sensitive_point, non_sensitive_point = 0, 0
    kernel_groups = []
    layer_sol = []
    point = 0

    for i in range(len(configs.sensitive_layer_kernel + configs.non_sensitive_layer_kernel)):
        if i in configs.sensitive_layer_idx:
            nk = configs.sensitive_layer_kernel[sensitive_point]
            ng = configs.sensitive_layer_group[sensitive_point]
            kg = np.array_split(np.zeros(nk), ng)
            kernel_groups.append(kg)
            layer_sol.append(solution[point: point + ng])
            point += ng
            sensitive_point += 1
        else:
            nk = configs.non_sensitive_layer_kernel[non_sensitive_point]
            ng = configs.non_sensitive_layer_group[non_sensitive_point]
            kg = np.array_split(np.zeros(nk), ng)
            kernel_groups.append(kg)
            layer_sol.append(solution[point: point + ng])
            point += ng
            non_sensitive_point += 1
    assert (np.concatenate(layer_sol, axis=0) == solution).all()

    if len(layer_sol) < len(configs.sorted_kernel_idx):
        assert configs.model_name == 'rescnn'
        layer_sol.insert(1, layer_sol[2])
        layer_sol.insert(5, layer_sol[6])
        layer_sol.insert(9, layer_sol[10])
        kernel_groups.insert(1, kernel_groups[2])
        kernel_groups.insert(5, kernel_groups[6])
        kernel_groups.insert(9, kernel_groups[10])
        assert (layer_sol[1] == layer_sol[3]).all()
        assert (layer_sol[5] == layer_sol[7]).all()
        assert (layer_sol[9] == layer_sol[11]).all()

    conv_info = _get_conv_info(layer_sol, kernel_groups, configs.sorted_kernel_idx)
    return conv_info


def _get_conv_info(layer_sol, kernel_groups, sorted_kernel_idx):
    """return {conv_idx: [active_kernel_idx, ...]}"""
    conv_info = {}
    assert len(layer_sol) == len(sorted_kernel_idx)
    for conv_idx in range(len(layer_sol)):
        each_conv_sorted_kernel_idx = np.array(sorted_kernel_idx[conv_idx])
        kg = kernel_groups[conv_idx]
        sol = layer_sol[conv_idx]
        sorted_active_kernel_idx = []
        for i in range(len(sol)):
            if sol[i] == 1:
                sorted_active_kernel_idx.append(kg[i] + 1)
            else:
                sorted_active_kernel_idx.append(kg[i])
        sorted_active_kernel_idx = np.concatenate(sorted_active_kernel_idx, axis=0)
        active_kernel_idx = each_conv_sorted_kernel_idx[np.where(sorted_active_kernel_idx != 0)]
        conv_info[f'conv_{conv_idx}'] = active_kernel_idx.tolist()
    return conv_info


def module_predict(module, dataset):
    outputs = []
    labels = []
    with torch.no_grad():
        for batch_inputs, batch_labels in dataset:
            batch_inputs = batch_inputs.to(device)
            batch_output = module(batch_inputs)
            outputs.append(batch_output)
            labels.append(batch_labels.to(device))
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
    return outputs, labels


def evaluate_ensemble_modules(modules, dataset):
    labels = None
    labels_for_check = None
    modules_outputs = []
    for target_class, m in enumerate(tqdm(modules, desc='modules', ncols=100)):
        each_module_outputs, labels = module_predict(m, dataset)
        # check
        if labels_for_check is not None:
            assert (labels_for_check == labels).all()
        else:
            labels_for_check = labels
        modules_outputs.append(each_module_outputs[:, target_class].unsqueeze(-1))
    modules_outputs = torch.cat(modules_outputs, dim=1)
    predicts = torch.argmax(modules_outputs, dim=1)
    acc = torch.mean((predicts == labels).float())
    return acc.cpu().item()


def load_population(generation, sol_dir, num_classes):
    populations = []
    for i in range(num_classes):
        path = f'{sol_dir}/gen_{generation}_exp_{i}_pop.pkl'
        with open(path, 'rb') as f:
            pop = pickle.load(f)
        populations.append(pop)
    return populations


def load_modules(configs, return_trained_model=False):
    trained_entire_model_path = f'{configs.trained_model_dir}/{configs.trained_entire_model_name}'
    model = load_model(configs.model_name, num_classes=configs.num_classes)
    model.load_state_dict(torch.load(trained_entire_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    populations = load_population(generation=configs.best_generation, sol_dir=configs.ga_save_dir,
                                  num_classes=configs.num_classes)
    modules = []
    for target_class in range(configs.num_classes):
        configs.set_sorted_kernel_idx(target_class)
        sol = populations[target_class][configs.best_sol_ensemble[target_class], :]
        conv_info = decode(sol, configs)
        each_module, active_kernel_idx = extract_module(conv_info, model)
        modules.append((each_module, active_kernel_idx))
    if return_trained_model:
        return modules, model
    else:
        return modules


def fuse_modules(module_kernels_1, module_kernels_2, entire_trained_model):
    fusion_kernels = set(module_kernels_1) | set(module_kernels_2)
    fusion_conv_info = {}
    for each_kernel in fusion_kernels:
        r = re.match(r'conv_(.+)_k_(.+)', each_kernel)
        conv_idx, kernel_idx = int(r.group(1)), int(r.group(2))
        temp = fusion_conv_info.get(f'conv_{conv_idx}', [])
        temp.append(kernel_idx)
        fusion_conv_info[f'conv_{conv_idx}'] = temp

    fusion_module, _ = extract_module(fusion_conv_info, entire_trained_model)
    return fusion_module


def load_range_of_module_output(module_output_path, mode, args=None):
    with open(module_output_path, 'rb') as f:
        all_module_outputs = pickle.load(f)
    if mode == 'min_max':
        ranges = [(min(outputs), max(outputs)) for outputs in all_module_outputs]
    elif mode == 'percentile':
        m, n = 10, 90
        if args is not None:
            m, n = args
            print(m, n)
        ranges = [(np.percentile(outputs, m), np.percentile(outputs, n)) for outputs in all_module_outputs]
    elif mode == 'outlier':
        ranges = []
        for each_module_outputs in all_module_outputs:
            q1 = np.percentile(each_module_outputs, 25)
            q3 = np.percentile(each_module_outputs, 75)
            upper_limit = q3 + 1.5 * (q3 - q1)
            lower_limit = q1 - 1.5 * (q3 - q1)
            ranges.append((lower_limit, upper_limit))
    elif mode == 'norm':
        ranges = [(np.mean(outputs), np.std(outputs)) for outputs in all_module_outputs]
    else:
        raise ValueError
    return ranges