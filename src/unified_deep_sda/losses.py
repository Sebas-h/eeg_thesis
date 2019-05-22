import torch as th
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class CCSALoss(th.nn.Module):
    """
    Classification and contrastive semantic alignment (CCSA) loss
    based on:
    """

    def __init__(self, alpha=0.5, margin=1.0):
        super(CCSALoss, self).__init__()
        self.alpha = alpha  # loss = (1-alpha)classification_loss + (alpha)CSA
        self.margin = margin
        self.eps = 1e-9

    def forward(self, outputs, target_vars):
        """
        Computes classification and contrastive semantic alignment loss
        :param outputs: outputs of forward pass model
        :param target_vars: true label value for given training data
        :return:
        """
        target_embedding = outputs['target_embedding']
        source_embedding = outputs['source_embedding']
        source_cls_pred_y = outputs['source_cls']
        source_true_y = target_vars[:, 1]

        # Compare pairs by labels: 1 where pair labels are the same, 0 otherwise
        gamma = th.eq(target_vars[:, 0], target_vars[:, 1])
        indices_same_label_pairs = gamma.nonzero()
        indices_different_label_pairs = (1 - gamma).nonzero()

        # Compute differences and split same/different samples
        diff = source_embedding - target_embedding
        ddist = diff.pow(2).sum(dim=[x for x in range(diff.dim()) if x != 0]).sqrt()
        diff = (ddist - ddist.min()) / (ddist.max() - ddist.min())

        diff_same = diff[indices_same_label_pairs]
        diff_diff = diff[indices_different_label_pairs]
        diff_same = diff_same.squeeze()
        diff_diff = diff_diff.squeeze()

        sa_loss = 0
        cs_loss = 0

        # todo: create batch generator that is balanced!
        #  These 'ifs' are a hacky solution for batches with just one kind of pair,
        #  which ideally should not happen.
        if diff_same.shape[0] > 0:
            sa_loss = diff_same
            # Euclidean distances (L2 norm)
            # dist_same = diff_same.pow(2).sum(dim=[x for x in range(diff_same.dim()) if x != 0]).sqrt()
            # Normalize distances (minmax), to balance with classification loss
            # dist_same = (dist_same - dist_same.min()) / (dist_same.max() - dist_same.min())
            # Compute semantic alignment loss
            # sa_loss = th.pow(dist_same, 2)
            sa_loss = sa_loss.mean()

        if diff_diff.shape[0] > 0:
            dist_diff = diff_diff
            # Euclidean distances (L2 norm)
            # dist_diff = diff_diff.pow(2).sum(dim=[x for x in range(diff_diff.dim()) if x != 0]).sqrt()
            # Normalize distances (minmax), to balance with classification loss
            # dist_diff = (dist_diff - dist_diff.min()) / (dist_diff.max() - dist_diff.min())
            # Compute class seperation loss
            cs_loss = self.margin - dist_diff
            cs_loss = th.clamp(cs_loss, min=0.0)
            # cs_loss = th.pow(cs_loss, 2)
            cs_loss = cs_loss.mean()

        # Contrastive loss:
        contrastive_loss = 0.5 * (sa_loss + cs_loss)

        # Classification loss:
        cls_loss = F.nll_loss(source_cls_pred_y, source_true_y)

        # Final weighted loss (weights in order to balance cls vs contrastive losses)
        loss = ((1 - self.alpha) * cls_loss) + (self.alpha * contrastive_loss)

        # Log/print individual loss values
        print(f"contrastive_loss={contrastive_loss}\tcls_loss={cls_loss}")
        return loss


if __name__ == '__main__':
    import pickle
    with open('losscalc.pickle', 'rb') as f:
        oo, tt = pickle.load(f)
    ccs = CCSALoss()
    print(ccs.forward(oo, tt))


##################################################
# OLD SCRAP CODE:
##################################################
# Minmax normalize embeddings
# for idx in range(batch_size):
#     min_val = target_embedding[idx].min()
#     max_val = target_embedding[idx].max()
#     target_embedding[idx] = (target_embedding[idx] - min_val) / (max_val - min_val)
#     min_val = source_embedding[idx].min()
#     max_val = source_embedding[idx].max()
#     source_embedding[idx] = (source_embedding[idx] - min_val) / (max_val - min_val)

# l = [0.98, .95, .96]
# nl = []
# for i in l:
#     nl.append((i - min(l)) / (max(l) - min(l)))
#
# print(l)
# print(nl)
# exit()

# a = np.array([1,2,3,4,11,22,4,2])
# print(a.mean())
# n = 5
# print(a[:n], a[:n].mean())
# print(a[n:], a[n:].mean())
# print(np.array([a[:n].mean(), a[n:].mean()]).mean())
# exit()

# input1 = source_embedding[indices_same_label_pairs].squeeze()
# input2 = target_embedding[indices_same_label_pairs].squeeze()
# print(input1.shape)
#
# cos = th.nn.CosineSimilarity(dim=1, eps=1e-6)
# output = cos(input1, input2)
# print(output.shape)
# print(th.mean(output))
# print()

# pdist = th.nn.PairwiseDistance(p=2)
# pwoutput = pdist(source_embedding, target_embedding)
# print(pwoutput)
# print(pwoutput.shape)

# exit()
#
# # euclidian distance OG
# diff = source_embedding - target_embedding
# dist = diff.pow(2).sum(dim=[x for x in range(diff.dim()) if x != 0]).sqrt()
# # dist = diff.pow(2).sum(dim=1).sqrt()
# # dist_sq = th.sum(th.pow(diff, 2), 1)
# # dist = th.sqrt(dist_sq)
#
# # Normalization (minmax normalization)
# sa_loss = dist[indices_same_label_pairs].squeeze()
# # print(sa_loss.mean())
# # sa_loss = gamma.float() * dist.float()
# # print(sa_loss.mean())
# sa_loss = (sa_loss - sa_loss.min()) / (sa_loss.max() - sa_loss.min())
# sa_loss = th.pow(sa_loss, 2)
#
# cs_loss = dist[indices_different_label_pairs].squeeze()
# # cs_loss = (1 - gamma).float() * dist.float()
# cs_loss = (cs_loss - cs_loss.min()) / (cs_loss.max() - cs_loss.min())
# cs_loss = self.margin - cs_loss
# cs_loss = th.clamp(cs_loss, min=0.0)
# cs_loss = th.pow(cs_loss, 2)
#
# # loss = dist_sq[indices_same_label_pairs] + th.pow(dist[indices_different_label_pairs], 2)
# # loss_alt = 0.5 * th.cat((sa_loss, cs_loss))
# loss_alt = 0.5 * th.cat((sa_loss, cs_loss))
# print(loss_alt)
# print(loss_alt.mean())
#
# # loss = 0.5 * (sa_loss.add(cs_loss))
# loss = 0.5 * (sa_loss.mean() + cs_loss.mean())
# print(loss)
# # print(loss.mean())
#
# # print(th.cat((sa_loss, cs_loss), dim=0).mean())
#
# exit()
#
# dist_sq = th.sum(th.pow(source_embedding - target_embedding, 2), 1)
# pwoutput = th.sqrt(dist_sq)
# print(pwoutput)
#
# # try power before normalization?
# pwoutput = pwoutput.pow(2)
#
# npwout = (pwoutput - pwoutput.min()) / (pwoutput.max() - pwoutput.min())
# print(npwout)
#
# # npwout = npwout.pow(2)
# # npwout *= 0.5
# print(f"loss: {npwout.mean()}")
#
# exit()
#
# # reshape embeddings (i.e. flatten)
# target_embedding = target_embedding.reshape((60, 560))
# source_embedding = source_embedding.reshape((60, 560))
#
# input1 = source_embedding[indices_same_label_pairs].squeeze()
# input2 = target_embedding[indices_same_label_pairs].squeeze()

# Gives 1 for very similar (i.e. 0 distance), 0 for dissimilar (i.e. 1 (max) distance)
# cos = th.nn.CosineSimilarity(dim=1, eps=1e-6)
# output = cos(input1, input2)
# print(output.shape)
# print(output.mean())
#
# exit()

# pdist = th.nn.PairwiseDistance(p=2)
# pwoutput = pdist(input1, input2)
# print(pwoutput)
#
# npwout = (pwoutput - pwoutput.min()) / (pwoutput.max() - pwoutput.min())
# print(npwout)
#
# exit()
#
# distances = th.norm((source_embedding - target_embedding), dim=1)

# distances = th.norm((source_embedding - target_embedding), dim=1)
# print(distances.shape)
# print(distances.sum())

# dd = (source_embedding - target_embedding).reshape((60, 560)).pow(2).sum(1)[indices_same_label_pairs]
# distances_of_same = distances[indices_same_label_pairs].squeeze()
# distances_of_different = distances[indices_different_label_pairs].squeeze()

# print(distances_of_same.sum())

# MinMax normalization of the distances:
# distances_of_same_normalized = (distances_of_same - distances_of_same.min()) / (
#         distances_of_same.max() - distances_of_same.min() + self.eps)
# distances_of_different_normalized = (distances_of_different - distances_of_different.min()) / (
#         distances_of_different.max() - distances_of_different.min() + self.eps)
# distances_of_same_normalized = distances_of_same
# distances_of_different_normalized = distances_of_different
# # print(distances_of_different_normalized)
# # distances_of_same_normalized = F.normalize(distances_of_same, dim=0)
# # distances_of_different_normalized = F.normalize(distances_of_different, dim=0)
#
# # print(distances_of_different_normalized)
# mdist = self.margin - distances_of_different_normalized
# # print(mdist)
# ddist = th.clamp(mdist, min=0.0)
# print(ddist)
# contrastive_loss = 0.5 * (distances_of_same_normalized.pow(2) + ddist.pow(2))
#
# print(contrastive_loss)
# mean_loss = contrastive_loss.mean()
# print(mean_loss)
# exit()

# print(distances_of_same.shape)
# print(distances_of_same.sum())
# print(distances_of_different.sum())

# euclidean distances
# distances = (source_embedding.reshape((60, 560)) - target_embedding.reshape((60, 560))).pow(2).sum(1).sqrt()
# print(distances.shape)
# print(distances.sum())

# normalize
# tensor_vec = distances
# tensor_vecnorm = (tensor_vec - tensor_vec.min()) / (tensor_vec.max() - tensor_vec.min())
# print(tensor_vecnorm)

# print(r.shape)
# print(r)
# print(source_embedding - target_embedding)
# s = gamma.unsqueeze(dim=1)
# ss = s.repeat((1, 560))
# inter = gamma.unsqueeze(dim=1).repeat((1, 560)).float() * r.float()
# print(f"inter shape: {inter.shape}")
#
# norms = th.norm(r.float(), dim=1)
# print(norms)
#
# exit()
#
# intertwo = (1 - gamma).unsqueeze(dim=1).repeat((1, 560)).float() * r.float()
# normstwo = th.norm(intertwo, dim=1)
# print(intertwo)
# print(normstwo)

# print(target_vars[:, 0])
# print(target_vars[:, 1])
# print(gamma)
# print(np.count_nonzero(gamma) / source_embedding.shape[0])

# Classification loss
# cls_loss = F.nll_loss(source_cls_pred_y, source_true_y)
#
# same_dist = []
# diff_dist = []
# for i in range(gamma.shape[0]):
#     if gamma[i] == 1:
#         same_dist.append(0.5 * th.norm(source_embedding[i] - target_embedding[i]).pow(2))
#     elif gamma[i] == 0:
#         diff_dist.append(
#             th.clamp(self.margin - (0.5 * th.norm(source_embedding[i] - target_embedding[i]).pow(2)),
#                      min=0.0))
# tot_dist = same_dist + diff_dist
#
# norm_dists = []
# for i in range(len(same_dist)):
#     norm_dists.append(same_dist[i] - min(same_dist) / (max(same_dist) - min(same_dist)))
#     norm_dists.append(diff_dist[i] - min(diff_dist) / (max(diff_dist) - min(diff_dist)))
#
# tot_norm_dist = sum(norm_dists)
# mean_tot_norm_dist = sum(norm_dists) / len(norm_dists)
# tot_norm_dist = th.mean(norm_dists)

# Constrastive loss, input.view(input.size(0), -1)
# distances = (source_embedding - target_embedding).pow(2).sum(1)  # squared distances
# src = source_embedding.view(-1)
# mm = th.mul(gamma, src)
# dist1 = th.norm(
#     gamma * (source_embedding.view(source_embedding.size(0), -1) -
#              target_embedding.view(target_embedding.size(0), -1))
# )
# dis2 = th.norm(
#     inverse_gamma * source_embedding - inverse_gamma * target_embedding
# )
# ll = dist1 + dis2
# losses = 0.5 * ((gamma * dist1) +
#                 (1 + -1 * gamma * F.relu(self.margin - (dist1 + self.eps).sqrt()).pow(2)))
#
# return cls_loss + losses

# Semantic alignment loss (distance measure):
# sa_loss = 0

# Class separation loss (similarity measure):
# cs_loss = 0

# euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
#
# loss_contrastive = th.mean(
#     ((1 - label) * th.pow(euclidean_distance, 2)) +
#     (label * th.pow(th.clamp(self.margin - euclidean_distance, min=0.0), 2))
# )
# return loss_contrastive

# Compute weighted (by alpha) loss
# loss = ((1 - self.alpha) * cls_loss) + (self.alpha * (sa_loss + cs_loss))
# return loss