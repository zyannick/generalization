
from default_launcher import *




class EpiFCRLauncher(DefaultLauncher):

    def __init__(self, algorithm_dict, algorithm_class, flags, hparams, input_shape, datasets, class_idx,
                 checkpoint_path, class_balance):
        super(EpiFCRLauncher, self).__init__(algorithm_dict, algorithm_class, flags, hparams, input_shape, datasets, class_idx,
                                             checkpoint_path, class_balance)
        self.train_dataloaders = {}
        self.algorithm_dict = algorithm_dict
        self.algorithm_name = algorithm_class
        self.class_balance = class_balance
        self.input_shape = input_shape
        self.class_idxs = class_idx
        self.flags = flags
        self.num_classes = flags.num_classes
        self.num_domains = flags.num_domains
        self.hparams = hparams
        self.datasets = datasets
        self.checkpoint_path = checkpoint_path
        self.setup_path()
        self.save_epoch_fmt_task = os.path.join(self.checkpoint_path,
                                                'checkpoint_{}ep.pt')
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.algorithm_class = get_algorithm_class(self.algorithm_name)
        self.algorithm = self.algorithm_class(flags, hparams, input_shape, datasets, class_idx, checkpoint_path,
                                              class_balance)
        if self.algorithm_dict is not None:
            self.algorithm.load_state_dict(algorithm_dict)
        self.algorithm.to(self.device)
        self.checkpoint_values = collections.defaultdict(lambda: [])
        self.iteration = 0
        self.best_accuracy_val = -1.0

    def setup_path(self):
        DefaultLauncher.setup_path(self)
        self.candidates = np.arange(0, len(self.train_dataloaders.keys()))

    def train_workflow(self):

        flags = self.flags

        self.algorithm.train()
        self.algorithm.ds_nn.train()
        self.algorithm.agg_nn.train()

        self.train_iters = {}
        for domain_key in self.train_dataloaders.keys():
            self.train_iters[domain_key] = iter(self.train_dataloaders[domain_key])

        last_results_keys = None

        while self.iteration < flags.num_iterations:
            step_start_time = time()
            step_vals = {}
            inputs_all = []
            labels_all = []
            domains_all = []

            for _, domain_key in enumerate(list(self.train_iters.keys())):
                inputs, labels, domains = next(self.train_iters[domain_key])
                inputs, labels, domains = Variable(inputs, requires_grad=False).cuda(),\
                                          Variable(labels, requires_grad=False).long().cuda() ,\
                                          Variable(domains, requires_grad=False).long().cuda()
                inputs_all.append(inputs)
                labels_all.append(labels)
                domains_all.append(domains)
            if self.iteration <= flags.loops_warm:
                ds_nn_values = self.algorithm.update_ds_nn(inputs_all, labels_all, domains_all)
                if flags.warm_up_agg == 1 and self.iteration <= flags.loops_agg_warm:
                        agg_nn_wanm_values = self.algorithm.update_agg_nn_warm(inputs_all, labels_all, domains_all)
            else:
                ds_nn_values = self.algorithm.train_ds_nn(inputs_all, labels_all, domains_all)
                agg_nn_wanm_values = self.algorithm.train_agg_nn(inputs_all, labels_all, domains_all)

            self.algorithm.iteration += 1
            self.iteration += 1

            self.checkpoint_values['step_time'].append(time() - step_start_time)
            for key, val in step_vals.items():
                self.checkpoint_values[key].append(val)

            if (self.iteration % self.flags.test_every == 0) or (self.iteration == flags.n_epochs - 1):
                results = {
                    'step': self.iteration,
                    'epoch': self.iteration / self.flags.steps_per_epoch,
                }

                for key, val in self.checkpoint_values.items():
                    results[key] = np.mean(val)

                result_dict, result_dict_per_domain = self.test_workflow()

                for domain_key in self.test_dataloaders.keys():
                    results[domain_key + '_acc'] = result_dict_per_domain['target'][domain_key]['accuracy']

                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    commons.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                commons.print_row([results[key]
                                   for key in results_keys], colwidth=12)

                results.update({
                    'hparams': self.hparams,
                    'args': vars(self.flags)
                })

                epochs_path = os.path.join(
                    self.checkpoint_path, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                # self.algorithm_dict = self.algorithm.state_dict()
                self.checkpoint_values = collections.defaultdict(lambda: [])

                if self.current_epoch % self.flags.save_every:
                    self.checkpointing()

    def test_workflow(self):

        self.source_iters = {}
        for domain_key in self.val_dataloaders.keys():
            self.source_iters[domain_key] = iter(
                self.val_dataloaders[domain_key])
        self.target_iter = iter(self.test_dataloaders)

        self.test_workflow_iters = {
            'source': self.source_iters,
            'target': self.target_iter
        }

        result_dict = {}
        result_dict_per_domain = {}

        result_dict['source'], result_dict_per_domain['source'] = self.test('source')

        if result_dict['source']['accuracy'] > self.best_accuracy_val:
            self.best_accuracy_val = result_dict['source']['accuracy']
            result_dict['target'], result_dict_per_domain['target'] = self.test('target')

        return result_dict, result_dict_per_domain



    def test(self, source_target):
        n_total = 0
        n_correct = 0

        list_task_predictions = []
        list_task_targets = []

        metrics_results_per_domain = {}

        with torch.no_grad():

            for domain_key in self.test_workflow_iters[source_target].keys():

                list_task_predictions_per_domain = []
                list_task_targets_per_domain = []

                for inputs, labels, domains in self.test_workflow_iters[source_target][domain_key]:

                    n_total += inputs.size(0)

                    inputs, labels, domains = Variable(inputs, requires_grad=False).cuda(), \
                                              Variable(labels, requires_grad=False).long().cuda(), \
                                              Variable(domains, requires_grad=False).long().cuda()

                    task_predictions, task_targets = self.algorithm.predict( inputs, labels, domains)

                    n_correct += task_predictions.eq(task_targets.data.view_as(task_predictions)).cpu().sum() / task_targets.size(2)

                    list_task_predictions_per_domain.extend( task_predictions.tolist())
                    list_task_targets_per_domain.extend(task_targets.tolist())

                metrics_results_per_domain[domain_key] = {
                    'task': {
                        'accuracy': accuracy_score(y_true=list_task_targets_per_domain, y_pred=list_task_predictions_per_domain),
                        'f1_score': f1_score(y_true=list_task_targets_per_domain, y_pred=list_task_predictions_per_domain, average='macro')
                    }
                }

                list_task_predictions.extend(list_task_predictions_per_domain)
                list_task_targets.extend(list_task_targets_per_domain)

        acc = n_correct * 1.0 / n_total

        metrics_results = {
            'task': {
                'accuracy': accuracy_score(y_true=list_task_targets, y_pred=list_task_predictions),
                'f1_score': f1_score(y_true=list_task_targets, y_pred=list_task_predictions, average='macro')
            }
        }


        if self.writer is not None:
            self.writer.add_histogram(
                'Test/' + source_target, task_predictions, self.current_epoch)
            self.writer.add_scalar(
                'Test/' + source_target + '_accuracy', acc, self.current_epoch)

        return metrics_results, metrics_results_per_domain

    def extract_features(self):
        raise NotImplementedError

    def vizualize_features(self):
        raise NotImplementedError
