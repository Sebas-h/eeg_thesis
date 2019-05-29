import numpy as np
import torch as th
import time
import braindecode.torch_ext.util as th_ext_util
import pandas as pd
from collections import OrderedDict


class SiameseTrainModel:
    def __init__(self, train_set, valid_set, test_set, model, optimizer, iterator, loss_function, stop_criterion,
                 model_constraint, cuda, func_compute_pred_labels, target_finetune_cls):
        # Config
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.model = model
        self.optimizer = optimizer
        self.iterator = iterator
        self.loss_function = loss_function
        self.stop_criterion = stop_criterion
        self.model_constraint = model_constraint
        self.func_compute_pred_labels = func_compute_pred_labels
        self.cuda = cuda
        self.target_finetune_cls = target_finetune_cls
        # Results
        self.epochs_df = pd.DataFrame()
        self.test_result = OrderedDict()

    def run(self):
        # Set cuda:
        if self.cuda:
            assert th.cuda.is_available(), "Cuda not available"
            self.model.cuda()
        # Train and monitor/evaluate model until stop
        while not self.stop_criterion.should_stop:
            self._run_one_epoch()

        # Load the last checkpoint with the best model, b/c of potential early stop
        # self.model.load_state_dict(th.load('checkpoint.pt'))
        self.model.load_state_dict(self.stop_criterion.checkpoint)

        # Final evalulation
        if self.test_set is not None:
            results = self._eval_epoch([('test', self.test_set)])
            # Save result
            self.test_result = results

    def _run_one_epoch(self):
        start = time.time()
        # Train:
        for batch_X, batch_y in self.iterator.get_batches(self.train_set, shuffle=True):
            self._train_batch(batch_X, batch_y)
        # Evaluate:
        results = self._eval_epoch((('train', self.train_set), ('valid', self.valid_set)))
        # Save epoch result:
        results.update(dict(runtime=time.time() - start))
        self.epochs_df = self.epochs_df.append(results, ignore_index=True)
        # Log:
        print(f'done epoch {self.epochs_df.shape[0]}')
        print(self.epochs_df.iloc[-1], '\n')

    def _train_batch(self, inputs, targets):
        """
        Train on given inputs and targets.
        :param inputs: `torch.autograd.Variable`
        :param targets: `torch.autograd.Variable`
        :return:
        """
        self.model.train()
        input_vars = th_ext_util.np_to_var(inputs)
        target_vars = th_ext_util.np_to_var(targets)
        if self.cuda:
            input_vars = input_vars.cuda()
            target_vars = target_vars.cuda()
        self.optimizer.zero_grad()
        if self.target_finetune_cls:
            outputs = self.model(input_vars, 'train', target_finetune_cls=True)
        else:
            outputs = self.model(input_vars, 'train')
        loss = self.loss_function(outputs, target_vars)
        loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)

    def _eval_epoch(self, setname_dataset_tuple):
        self.model.eval()
        epoch_results = OrderedDict()
        for setname, dataset in setname_dataset_tuple:
            # Collect all predictions and losses
            all_preds = []
            all_losses = []
            batch_sizes = []
            for batch_X, batch_y in self.iterator.get_batches(dataset, shuffle=False):
                preds, loss = self._eval_batch(batch_X, batch_y, setname)
                all_preds.append(th_ext_util.var_to_np(preds))
                all_losses.append(loss)
                batch_sizes.append(len(batch_X))

            # Compute mean per-input loss
            batch_weights = np.array(batch_sizes) / float(np.sum(batch_sizes))
            loss_per_batch = [np.mean(loss) for loss in all_losses]
            mean_loss = np.sum(batch_weights * loss_per_batch)

            # Compute predictions and accuracy/inverse_of_error
            predicted_labels = self.func_compute_pred_labels(all_preds, dataset)
            if len(dataset.y.shape) > 1:  # for Siamase network (paired data) todo clean up
                accuracy = np.mean(predicted_labels == dataset.y[:, 1])
            else:
                accuracy = np.mean(predicted_labels == dataset.y)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if setname == 'valid':
                self.stop_criterion(mean_loss, self.model)

            # Save results
            epoch_results.update({
                f'{setname}_loss': mean_loss,
                f'{setname}_accuracy': accuracy
            })
        return epoch_results

    def _eval_batch(self, inputs, targets, setname):
        net_in = th_ext_util.np_to_var(inputs)
        net_target = th_ext_util.np_to_var(targets)
        if self.cuda:
            net_in = net_in.cuda()
            net_target = net_target.cuda()

        if self.target_finetune_cls:
            outputs = self.model(net_in, setname, target_finetune_cls=True)
        else:
            outputs = self.model(net_in, setname)

        loss = self.loss_function(outputs, net_target)
        loss = float(th_ext_util.var_to_np(loss))
        if type(outputs) == dict:  # for Siamase network (paired data) todo clean up
            outputs = outputs['source_cls']
        return outputs, loss
