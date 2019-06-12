import numpy as np
import time
import braindecode.torch_ext.util as th_ext_util
import pandas as pd
from collections import OrderedDict
import logging

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_set, valid_set, test_set, model, optimizer,
                 iterator, loss_function, stop_criterion,
                 model_constraint, cuda, func_compute_pred_labels,
                 siamese):
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
        self.siamese = siamese
        # Results
        self.epochs_df = pd.DataFrame()
        self.test_result = OrderedDict()

    def train(self):
        # Set cuda
        if self.cuda:
            self.model.cuda()

        # Train and evaluate models until stop
        while not self.stop_criterion.should_stop:
            self._train_one_epoch()

        # Load the last checkpoint with the best models,
        #   b/c of potential early stop
        self.model.load_state_dict(self.stop_criterion.checkpoint)

        # Final evalulation
        if self.test_set is not None:
            results = self._eval_epoch([('test', self.test_set)])
            # Save result
            self.test_result = results

    def _train_one_epoch(self):
        start = time.time()
        # Train:
        for batch_X, batch_y in self.iterator.get_batches(self.train_set,
                                                          shuffle=True):
            self._train_batch(batch_X, batch_y)

        # Evaluate:
        results = self._eval_epoch(
            (
                ('train', self.train_set),
                ('valid', self.valid_set),
                ('test', self.test_set)
            )
        )

        # Save epoch result:
        results.update(dict(runtime=time.time() - start))
        self.epochs_df = self.epochs_df.append(results, ignore_index=True)

        # Log:
        print(f'done epoch {self.epochs_df.shape[0]}')
        cols = ["runtime", "train_accuracy", "valid_accuracy", "train_loss",
                "valid_loss"]
        if self.test_set is not None:
            cols.insert(3, 'test_accuracy')
            cols.insert(len(cols), 'test_loss')
        print(self.epochs_df.iloc[-1][cols], '\n')

    def _train_batch(self, inputs, targets):
        """
        Train on given inputs and targets.
        :param inputs: `torch.autograd.Variable`
        :param targets: `torch.autograd.Variable`
        :return:
        """
        self.model.train()
        input_vars = self.np_to_tensor(inputs)
        target_vars = self.np_to_tensor(targets)
        self.optimizer.zero_grad()
        outputs = self.model(input_vars)
        loss = self.loss_function(outputs, target_vars)
        loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)

    def _eval_epoch(self, setname_dataset_tuple):
        self.model.eval()
        epoch_results = OrderedDict()
        for setname, dataset in setname_dataset_tuple:
            if dataset is None:
                continue
            # Collect all predictions and losses
            all_preds = []
            all_losses = []
            batch_sizes = []
            for batch_X, batch_y in self.iterator.get_batches(dataset,
                                                              shuffle=False):
                batch_size = len(batch_X)
                preds, loss = self._eval_batch(batch_X, batch_y)
                if self.siamese:
                    preds = preds['cls']
                    batch_size = len(batch_X['source'])
                all_preds.append(th_ext_util.var_to_np(preds))
                all_losses.append(loss)
                batch_sizes.append(batch_size)

            # Compute mean per-input loss
            batch_weights = np.array(batch_sizes) / float(np.sum(batch_sizes))
            loss_per_batch = [np.mean(loss) for loss in all_losses]
            mean_loss = np.sum(batch_weights * loss_per_batch)

            # Compute predictions and accuracy/inverse_of_error
            predicted_labels = self.func_compute_pred_labels(all_preds, dataset)
            accuracy = np.mean(predicted_labels == dataset.y)
            if self.siamese:
                accuracy = np.mean(predicted_labels == dataset.y['source'])

            # early_stopping needs the validation loss
            #   to check if it has decresed, and if it has,
            #   it will make a checkpoint of the current models
            if setname == 'valid':
                self.stop_criterion(mean_loss, self.model)

            # Save results
            epoch_results.update({
                f'{setname}_loss': mean_loss,
                f'{setname}_accuracy': accuracy
            })

        return epoch_results

    def _eval_batch(self, inputs, targets):
        net_in = self.np_to_tensor(inputs)
        net_target = self.np_to_tensor(targets)
        outputs = self.model(net_in)
        loss = self.loss_function(outputs, net_target)
        loss = float(th_ext_util.var_to_np(loss))
        return outputs, loss

    def np_to_tensor(self, batch):
        if self.siamese:
            batch['source'] = th_ext_util.np_to_var(batch['source'])
            batch['target'] = th_ext_util.np_to_var(batch['target'])
            if self.cuda:
                batch['source'] = batch['source'].cuda()
                batch['target'] = batch['target'].cuda()
        else:
            batch = th_ext_util.np_to_var(batch)
            if self.cuda:
                batch = batch.cuda()
        return batch
