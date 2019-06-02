from braindecode.models import shallow_fbcsp


class TransferLearningOne:
    def __init__(self, source_data, target_data, model):
        self.source_data = source_data
        self.target_data = target_data
        self.model = model
        self.trained_source_model = None

    def run(self):
        self.train_target()
        self.train_source()

    def train_target(self, train_test_ratio=(0.7, 0.3)):
        train_set, test_set = [[], []]
        # train models
        # models()
        pass

    def train_source(self):
        pass


import numpy as np

# all_losses = [[2, 3, 2, 3], [1, 2, 3, 5], [0, 1, 2, 5]]
# batch_sizes = [58, 57, 57]
# # mean_losss = np.mean(np.array(all_losses) * np.array(batch_sizes) / np.mean(batch_sizes))
# # print(mean_losss)
# #
# # batch_weights = np.array(batch_sizes) / float(np.sum(batch_sizes))
# # loss_per_batch = [np.mean(loss) for loss in all_losses]
# # mean_loss = np.sum(batch_weights * loss_per_batch)
# # print(mean_loss)
# # print(np.mean(all_losses))
# #
# a = np.mean(all_losses, axis=1)
# print(a * [0.3, 0.5, 0.2])
#
# all_losses = np.array(all_losses).astype(float)
# b = np.array([all_losses[0] * 0.3, all_losses[1] * 0.5, all_losses[2] * 0.2])
# print(np.mean(b, axis=1))

x = np.array([[[0], [1], [2]]])
y = x.squeeze()

print(x)
print(y)
