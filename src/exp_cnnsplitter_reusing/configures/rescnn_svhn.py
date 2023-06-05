from exp_cnnsplitter_reusing.global_configure import GlobalConfigures
from exp_cnnsplitter_reusing.kernel_importance_loader import load_kernel_importance


class Configures(GlobalConfigures):
    def __init__(self):
        super(Configures, self).__init__()
        self.model_name = 'rescnn'
        self.dataset_name = 'svhn'
        self.num_classes = 10

        self.best_generation = 107
        self.best_sol_ensemble = [0, 0, 0, 22, 0, 0, 0, 0, 0, 0]
        self.log_idx = 'recorder_1'
        self.best_acc = f'93.52%'
        self.best_diff = f'60.46%'

        self.workspace = f'{self.data_dir}/{self.model_name}_{self.dataset_name}'
        self.trained_model_dir = f'{self.workspace}'
        self.kernel_importance_dir = f'{self.workspace}/kernel_importance'
        self.module_output_path = f'{self.workspace}/module_outputs_{self.log_idx}.pkl'
        self.ga_save_dir = f'{self.workspace}/ga'
        self.dataset_dir = f'{self.dataset_dir}/svhn'

        self.num_generations = 200  # Number of generations.
        self.num_parents_mating = 50  # Number of solutions to be selected as parents in the mating pool.
        self.num_sol_per_pop = 100  # Number of solutions in the population.
        self.keep_parents = 20  # Number of parents to keep in the next population. -1: keep all and 0: keep nothing.
        self.parent_selection_type = "sss"  # Type of parent selection.
        self.init_pop_mode = ['heuristic', 'random'][0]

        self.crossover_type = "single_point"  # Type of the crossover operator.
        self.mutation_type = "random"  # Type of the mutation operator.
        self.mutation_percent_genes = 10  # Percentage of genes to mutate.

        # conv_1, 5, 9 as same as conv_3, 7, 11 because of residual layer. so the decode() is special.
        conv_kernels = [64,
                        128, 128,
                        256,
                        512, 512,
                        512,
                        512, 512]
        sensitive_layer_idx = [2, 3, 5]  # conv_1, 3, 4, 5, 7 -> -, 2, 3, -, 5
        sensitive_layer_kernel = [conv_kernels[i] for i in sensitive_layer_idx]
        # sensitive_layer_group = [n_kernels for n_kernels in sensitive_layer_kernel]
        sensitive_layer_group = [10 if n_kernels < 256 else 100 for n_kernels in sensitive_layer_kernel]

        non_sensitive_layer_idx = [i for i in range(len(conv_kernels)) if i not in sensitive_layer_idx]
        non_sensitive_layer_kernel = [conv_kernels[i] for i in non_sensitive_layer_idx]
        # non_sensitive_layer_group = [n_kernels for n_kernels in non_sensitive_layer_kernel]
        non_sensitive_layer_group = [10 if n_kernels < 256 else 100 for n_kernels in non_sensitive_layer_kernel]

        self.sensitive_layer_idx = sensitive_layer_idx
        self.sensitive_layer_kernel = sensitive_layer_kernel
        self.sensitive_layer_group = sensitive_layer_group
        self.sensitive_layer_active_gene_ratio = [0.3, 0.4, 0.5]

        self.non_sensitive_layer_idx = non_sensitive_layer_idx
        self.non_sensitive_layer_kernel = non_sensitive_layer_kernel
        self.non_sensitive_layer_group = non_sensitive_layer_group
        self.non_sensitive_layer_active_gene_ratio = [0.2, 0.3]

        self.sorted_kernel_idx = None

        # alpha = alpha + max(0, gamma - acc)
        # fitness = alpha * acc + (1 - alpha) * diff
        self.alpha = 0.9
        self.gamma = 0.9  # according to model's accuracy
        self.acc_thresholds = {2: 0.95, 4: 0.95, 8: 0.9, 10: 0.85}  # for clip

        # communicate between explorers and the recorder
        self.signal_dir = f'{self.ga_save_dir}/signals'
        self.explorer_finish_signal_list = [f'{self.signal_dir}/explorer_{i}.sig' for i in range(10)]
        self.recorder_finish_signal_list = [f'{self.signal_dir}/recorder_{i}.sig' for i in range(10)]

    def set_sorted_kernel_idx(self, target_class):
        if self.kernel_importance_analyzer_mode == 'random':
            print(f'kernel_importance_analyzer_mode is RANDOM.')
            kernel_importance = load_kernel_importance(self.kernel_importance_analyzer_mode,
                                                       self.kernel_importance_dir)
            target_class_ki = kernel_importance[target_class]
            sorted_kernel_idx = []
            for each_conv_ki in target_class_ki:
                sorted_kernel_idx.append(list(range(len(each_conv_ki))))
            self.sorted_kernel_idx = sorted_kernel_idx
        elif self.kernel_importance_analyzer_mode == 'L1':
            # prepare kernel indices. sort by importance
            kernel_importance = load_kernel_importance(self.kernel_importance_analyzer_mode,
                                                       self.kernel_importance_dir)
            target_class_ki = kernel_importance[target_class]
            sorted_kernel_idx = []
            for each_conv_ki in target_class_ki:
                sorted_kernel_idx.append([
                    item[0] for item in sorted(enumerate(each_conv_ki), key=lambda x: x[1], reverse=True)
                ])
            self.sorted_kernel_idx = sorted_kernel_idx
        else:
            raise ValueError