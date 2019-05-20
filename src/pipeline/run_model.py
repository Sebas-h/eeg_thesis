import torch as th
from src.pipeline import pytorchtools
from src.pipeline.train_model import TrainModel
from src.pipeline.train_setup import TrainSetup
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.optimizers import AdamW
import pandas as pd
import datetime
import uuid
from src.unified_deep_sda.siamese_train_model import SiameseTrainModel
import torch.nn.functional as F

class RunModel:
    def __init__(self):
        pass

    @staticmethod
    def go(train_set, valid_set, test_set, n_classes, subject_id,
           tl_model_state=None,
           tl_freeze=False,
           tl_eegnetautoencoder=False,
           sda_freeze=False):

        ############################################
        # config
        ############################################
        # 'shallow' or 'deep' or 'eegnet'
        model_name = 'siamese_eegnet'
        # model_name = 'eegnet'
        # cropped or trialwise training
        cropped = False
        # cross validation yes or no
        cv = False
        # Transfer learning: all but one training method, min dtw dist
        tl_abo = False

        # Max number of epochs if early stopping criteria not satisfied:
        max_epochs = 900
        # Early stopping (patience) value:
        max_increase_epochs = 100
        # Number of training examples to train per optimization step (i.e. per batch):
        batch_size = 60
        # Cuda check
        cuda = th.cuda.is_available()

        # Optimizer parameters
        lr = 1e-3
        weight_decay = 0

        # if model_name == 'shallow':
        #     lr = 0.000625
        # elif model_name == 'deep':
        #     lr = 0.01
        #     weight_decay = 0.0005

        # Cropped training parameters
        input_time_length = 1000
        final_conv_length_shallow = 30
        final_conv_length_deep = 2
        ############################################
        ############################################

        ################################################################################################################
        # Create setup for model training based on given config parameters
        if model_name == 'siamese_eegnet' and tl_model_state is not None:
            train_setup = TrainSetup(
                cropped=cropped,
                train_set=train_set,
                model_name=model_name,
                cuda=cuda,
                batch_size=batch_size,
                n_classes=n_classes,
                input_time_length=input_time_length,
                final_conv_length_shallow=final_conv_length_shallow,
                final_conv_length_deep=final_conv_length_deep,
                sda_finetune=True
            )
        else:
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

        # Assign created setup
        model = train_setup.model
        iterator = train_setup.iterator
        loss_function = train_setup.loss_function
        func_compute_pred_labels = train_setup.compute_pred_labels_func

        # Log model config/design/architecture
        print(model)

        ################################################################################################################
        # Transfer learning:
        # todo: clean up and make (at least some) generic
        if tl_model_state is not None:
            if tl_eegnetautoencoder:
                ae_params_dict = th.load(tl_model_state)
                ae_params_values = list(ae_params_dict.values())[:-4]  # -4 for eegnet ae
                model_state_dict = model.state_dict()
                for idx, (param_name, value) in enumerate(model_state_dict.items()):
                    if idx > (len(ae_params_values) - 1):
                        break
                    if value.shape == ae_params_values[idx].shape:
                        model_state_dict[param_name] = ae_params_values[idx]
                model.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(th.load(tl_model_state))
            if tl_freeze:
                for idx, child in enumerate(model.named_children()):
                    # print(idx, child[0], [x.shape for x in child[1].parameters()])
                    if idx > 13:
                        break
                    for param in child[1].parameters():
                        param.requires_grad = False
            if sda_freeze:
                for idx, child in enumerate(model.named_children()):
                    if child[0] == 'embed':
                        for param in child[1].parameters():
                            param.requires_grad = False
        ################################################################################################################

        ################################################################################################################
        # Set up additional model training variables:
        stop_criterion = pytorchtools.EarlyStopping(patience=max_increase_epochs, verbose=False, max_epochs=max_epochs)
        model_constraint = MaxNormDefaultConstraint()
        # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # only include parameters that require grad (i.e. are not frozen)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        ################################################################################################################

        ################################################################################################################
        # Initialize trainable model
        if model_name == 'siamese_eegnet' and tl_model_state is not None:
            train_model = SiameseTrainModel(
                train_set=train_set,
                valid_set=valid_set,
                test_set=test_set,
                model=model,
                optimizer=optimizer,
                iterator=iterator,
                loss_function=F.nll_loss,
                stop_criterion=stop_criterion,
                model_constraint=model_constraint,
                cuda=cuda,
                func_compute_pred_labels=func_compute_pred_labels,
                target_finetune_cls=True
            )
        else:
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

        # Train model:
        train_model.run()

        # Log training perfomance
        print(train_model.epochs_df)

        # Log test perfomance
        print(train_model.test_result)

        # print([p for p in model.conv_temporal.parameters()])
        # print([p for p in model.conv_separable_point.parameters()])
        # print([p for p in model.conv_classifier.parameters()])

        # Generate unique UUID for save files and log it
        unique_id = uuid.uuid4().hex
        print("UUID:", unique_id)

        # Save results and model
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        file_name = f"{timestamp}_subject_{subject_id}_{unique_id}.csv"
        train_model.epochs_df.to_csv(file_name)

        # Save model state (parameters)
        file_name_state_dict = f'model_sate_subject_{subject_id}_{unique_id}.pt'
        th.save(model.state_dict(), file_name_state_dict)
        return file_name_state_dict


if __name__ == '__main__':
    from braindecode.models.eegnet import EEGNetv4


    def test_tl_eegnet_ae():
        params_dict = th.load("/Users/sebas/code/thesis/pipeline/model_sate_s26_deep.pt")
        params_values = list(params_dict.values())[:-4]
        model = EEGNetv4(22, 4, input_time_length=1125).create_network()
        model_dict = model.state_dict()

        for idx, (key, value) in enumerate(model_dict.items()):
            if idx > (len(params_values) - 1):
                break
            if value.shape == params_values[idx].shape:
                print(f"model {value.shape} == pretrained {params_values[idx].shape}")
                model_dict[key] = params_values[idx]

        print(model.state_dict())
        model.load_state_dict(model_dict)
        print('###########\n###########')
        print(model.state_dict())


    test_tl_eegnet_ae()
    print('done')
