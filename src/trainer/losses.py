import torch as th
import torch.nn.functional as F


def get_loss(config):
    model_name = config['model']['name']
    if model_name in ('siamese_eegnet', 'siamese_deep', 'siamese_shallow'):
        return CCSALoss(alpha=0.5)
    if model_name == 'eegnet_cae':
        return th.nn.MSELoss()
    if config['cropped']['use']:
        return lambda preds, targets: F.nll_loss(
            th.mean(preds, dim=2, keepdim=False), targets)
    return F.nll_loss


class CCSALoss(th.nn.Module):
    """
    Classification and contrastive semantic alignment (CCSA) loss
    based on:
    """
    def __init__(self, alpha=0.5, margin=1.0):
        super(CCSALoss, self).__init__()
        self.alpha = alpha  # loss = (1-alpha)cls_loss + (alpha)CSA
        self.margin = margin
        self.eps = 1e-9

    def forward(self, outputs, target_vars):
        """
        Computes classification and contrastive semantic alignment loss
        :param outputs: outputs of forward pass models
        :param target_vars: true label value for given training data
        :return:
        """
        target_embedding = outputs['target_embedding']
        source_embedding = outputs['source_embedding']
        source_cls_pred_y = outputs['source_cls']

        # todo: hacky stuff here; 1 for true y from soure, 0 for true y from target:
        # source_true_y = target_vars[:, 1]
        source_true_y = target_vars[:, 0]

        # Compare pairs by labels: 1 where pair labels are the same, 0 otherwise
        gamma = th.eq(target_vars[:, 0], target_vars[:, 1])
        indices_same_label_pairs = gamma.nonzero()
        indices_different_label_pairs = (1 - gamma).nonzero()

        # Euclidean distances (L2 norm) distances and normalization
        diff = source_embedding - target_embedding
        ddist = diff.pow(2).sum(
            dim=[x for x in range(diff.dim()) if x != 0]).sqrt()
        diff = (ddist - ddist.min()) / (ddist.max() - ddist.min())

        diff_same = diff[indices_same_label_pairs]
        diff_diff = diff[indices_different_label_pairs]
        # diff_same = diff_same.squeeze()
        # diff_diff = diff_diff.squeeze()

        sa_loss = 0
        cs_loss = 0

        # todo: create batch generator that is balanced!
        #  These 'ifs' are a hacky solution for batches
        #  with just one kind of pair, which ideally should not happen.
        if diff_same.shape[0] > 0:
            sa_loss = diff_same
            # sa_loss = th.pow(dist_same, 2)
            sa_loss = sa_loss.mean()

        if diff_diff.shape[0] > 0:
            dist_diff = diff_diff
            # Compute class seperation loss
            cs_loss = self.margin - dist_diff
            cs_loss = th.clamp(cs_loss, min=0.0)
            # cs_loss = th.pow(cs_loss, 2)
            cs_loss = cs_loss.mean()

        # Contrastive loss:
        contrastive_loss = 0.5 * (sa_loss + cs_loss)

        # Classification loss:
        cls_loss = F.nll_loss(source_cls_pred_y, source_true_y)

        # Final weighted loss (to balance cls vs contrastive loss)
        loss = ((1 - self.alpha) * cls_loss) + (self.alpha * contrastive_loss)

        return loss
