from braindecode.torch_ext.optimizers import AdamW


def get_optmizer(model, config):
    # only include parameters that require grad (i.e. are not frozen)
    return AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=config['optimizer']['lr'],
                 weight_decay=config['optimizer']['weight_decay'])
