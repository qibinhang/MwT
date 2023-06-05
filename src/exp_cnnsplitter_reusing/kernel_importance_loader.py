import os
import _pickle as pickle


def load_kernel_importance(mode, save_dir):
    if mode == 'L1':
        importance_path = f'{save_dir}/avg_L1.pkl'
    elif mode == 'random':
        importance_path = f'{save_dir}/avg_L1.pkl'
    else:
        raise ValueError

    if not os.path.exists(importance_path):
        raise IOError(f'{importance_path} not exists, '
                      f'please generate kernel_importance using kernel_importance_analyzer.py first.')

    with open(importance_path, 'rb') as f:
        kernel_importance = pickle.load(f)
    return kernel_importance