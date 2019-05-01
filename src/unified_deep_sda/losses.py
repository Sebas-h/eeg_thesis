import torch as th
import torch.nn.functional as F


class CCSALoss(th.nn.Module):
    """
    Classification and contrastive semantic alignment (CCSA) loss
    based on:
    """

    def __init__(self, alpha=0.25, margin=1.0):
        super(CCSALoss, self).__init__()
        self.alpha = alpha
        self.margin = margin

    def forward(self, outputs, target_vars):
        """

        :param outputs: outputs of forward pass model
        :param target_vars: true label value for given training data
        :return:
        """
        x_source = outputs[0]
        y_source_preds = outputs[1]
        x_target = outputs[2]

        # Classification loss
        cls_loss = F.nll_loss(y_source_preds, target_vars)

        # Semantic alignment loss (distance measure):
        sa_loss = 0

        # Class separation loss (similarity measure):
        cs_loss = 0

        # euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        #
        # loss_contrastive = th.mean(
        #     ((1 - label) * th.pow(euclidean_distance, 2)) +
        #     (label * th.pow(th.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # )
        # return loss_contrastive

        # Compute weighted (by alpha) loss
        loss = ((1 - self.alpha) * cls_loss) + (self.alpha * (sa_loss + cs_loss))
        return loss
