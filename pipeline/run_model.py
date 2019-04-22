import torch as th
import pytorchtools
from train_model import TrainModel
from train_setup import TrainSetup
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.optimizers import AdamW
import pandas as pd
import datetime


class RunModel:
    def __init__(self):
        pass

    def go(self, train_set, valid_set, test_set, n_classes, subject_id, tl_model_state=None, tl_freeze=False):
        ############################################
        # config
        ############################################
        # 'shallow' or 'deep' or 'eegnet'
        model_name = 'eegnet_cae'
        # cropped or trialwise training
        cropped = False
        # cross validation yes or no
        cv = False
        # Transfer learning: all but one training method, min dtw dist
        tl_abo = False

        # Max number of epochs if early stopping criteria not satisfied:
        max_epochs = 750
        # Early stopping (patience) value:
        max_increase_epochs = 100
        # Number of training examples to train per optimization step (i.e. per batch):
        batch_size = 60
        # Cuda check
        cuda = th.cuda.is_available()

        # Optimizer parameters
        lr = 1e-3
        weight_decay = 0

        # Cropped training parameters
        input_time_length = 1000
        final_conv_length_shallow = 30
        final_conv_length_deep = 2
        ############################################
        ############################################

        ################################################################################################################
        train_setup = TrainSetup(
            cropped=cropped,
            train_set=train_set,
            model_name=model_name,
            cuda=cuda,
            batch_size=batch_size,
            n_classes=n_classes,
            input_time_length=input_time_length,
            final_conv_length_shallow=final_conv_length_shallow,
            final_conv_length_deep=final_conv_length_deep
        )

        model = train_setup.model
        print(model)  # log model config/architecture

        # Transfer learning:
        if tl_model_state is not None:
            model.load_state_dict(th.load(tl_model_state))
            if tl_freeze:
                for idx, child in enumerate(model.named_children()):
                    # print(idx, child[0], [x.shape for x in child[1].parameters()])
                    if idx > 13:
                        continue
                    for param in child[1].parameters():
                        param.requires_grad = False

        iterator = train_setup.iterator
        loss_function = train_setup.loss_function
        func_compute_pred_labels = train_setup.compute_pred_labels_func
        stop_criterion = pytorchtools.EarlyStopping(patience=max_increase_epochs, verbose=False, max_epochs=max_epochs)
        model_constraint = MaxNormDefaultConstraint()
        # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

        ################################################################################################################

        ################################################################################################################
        # Train the model
        train_model = TrainModel(
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            loss_function=loss_function,
            stop_criterion=stop_criterion,
            model_constraint=model_constraint,
            cuda=cuda,
            func_compute_pred_labels=func_compute_pred_labels
        )
        train_model.run()
        print('Done')
        print(train_model.epochs_df)
        print(train_model.test_result)
        # print([p for p in model.conv_temporal.parameters()])
        # print([p for p in model.conv_separable_point.parameters()])
        # print([p for p in model.conv_classifier.parameters()])

        # Save results and model
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        file_name = f"{timestamp}_subject_{subject_id}.csv"
        train_model.epochs_df.to_csv(file_name)

        # Save model state (parameters)
        th.save(model.state_dict(), f'model_sate_s{subject_id}_deep.pt')
        ################################################################################################################
