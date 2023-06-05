def load_configure(model_name, dataset_name):
    import sys
    sys.path.append('..')

    model_dataset_name = f'{model_name}_{dataset_name}'
    if model_dataset_name == 'simcnn_cifar10':
        from configures.simcnn_cifar10 import Configures
    elif model_dataset_name == 'simcnn_svhn':
        from configures.simcnn_svhn import Configures
    elif model_dataset_name == 'rescnn_cifar10':
        from configures.rescnn_cifar10 import Configures
    elif model_dataset_name == 'rescnn_svhn':
        from configures.rescnn_svhn import Configures
    # elif model_dataset_name == 'simcnn_svhn_5':
    #     from configures.simcnn_svhn_5 import Configures
    # elif model_dataset_name == 'rescnn_svhn_5':
    #     from configures.rescnn_svhn_5 import Configures
    else:
        raise ValueError()
    configs = Configures()
    return configs