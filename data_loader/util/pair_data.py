import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.iterators import get_balanced_batches
import random


def create_paired_dataset(target_data, source_data, n_classes, seed=123456789):
    np.random.seed(seed)
    target_indices = []
    source_indices = [item for item in range(source_data.y.shape[0])
                      for _ in range(2)]

    # Keep track of how many times each target samples has been paired
    target_picked_counter = np.zeros((target_data.y.shape[0]))
    for i_src_sample in np.arange(source_data.y.shape[0]):
        label = source_data.y[i_src_sample]

        # Indices where target y has same label as current src sample:
        same = np.where(target_data.y == label)[0]

        # Among target samples viable to be picked,
        #   pick one that has been paired up least so far:
        m = np.min(target_picked_counter[same])
        is_same_pick_from = np.where(target_picked_counter[same] == m)[0]
        same_pick_from = same[is_same_pick_from]
        a = np.random.randint(0, same_pick_from.shape[0])
        i_random_same = same_pick_from[a]

        # Divide target into sets of samples with same label,
        #   but only those target samples with different label
        #   to current src sample
        diffs = []
        for lbl in [x for x in range(n_classes) if x != label]:
            # assumes class labels go from 0 to num_classes - 1
            diffs.append(
                np.where(target_data.y == lbl)[0]
            )
        # Select set from same-labeled-target-sample sets that has been
        #   paired up least so far, which for the target samples
        #   vialble to be picked from:
        lowest_i = 0
        lowest_val = np.inf
        for i, diff in enumerate(diffs):
            s = sum(target_picked_counter[diff])
            if s < lowest_val:
                lowest_val = s
                lowest_i = i
        diff = diffs[lowest_i]

        # Among target samples viable to be picked,
        #   pick one that has been paired up least so far:
        dmin = np.min(target_picked_counter[diff])
        is_diff_pick_from = np.where(target_picked_counter[diff] == dmin)[0]
        diff_pick_from = diff[is_diff_pick_from]
        b = np.random.randint(0, diff_pick_from.shape[0])
        i_random_diff = diff_pick_from[b]

        # Increase counter for picked target samples (indices)
        target_picked_counter[i_random_same] += 1
        target_picked_counter[i_random_diff] += 1

        # Remember chosen target indices to be paired
        target_indices.append(i_random_same)  # also called positive pair
        target_indices.append(i_random_diff)  # also called negative pair

    # Create paired data
    paired_dataset = create_pairs(source_data, target_data, source_indices,
                                  target_indices)

    return paired_dataset


def create_pairs(source_data, target_data, source_indices, target_indices):
    # Create list for faster indexing
    source_data.X = list(source_data.X)
    source_data.y = list(source_data.y)
    target_data.X = list(target_data.X)
    target_data.y = list(target_data.y)

    return SignalAndTarget(
        {
            'source': [source_data.X[i] for i in source_indices],
            'target': [target_data.X[i] for i in target_indices]
        },
        {
            'source': [source_data.y[i] for i in source_indices],
            'target': [target_data.y[i] for i in target_indices]
        }
    )


def split_paired_into_train_test(paired_dataset, n_folds, i_test_fold,
                                 n_classes,
                                 rng=None):
    # Indexing with lists faster than ndarrays:
    assert type(paired_dataset.X['source']) == list and \
           type(paired_dataset.X['source']) == list, \
           "Expected paired dataset X to be list containing ndarrays."

    n_trials = len(paired_dataset.X['source'])
    if n_trials < n_folds:
        raise ValueError("Less Trials: {:d} than folds: {:d}".format(
            n_trials, n_folds
        ))

    train_indices = []
    test_indices = []

    source_y = paired_dataset.y['source']
    target_y = paired_dataset.y['target']

    n_possible_label_pairs = n_classes ** 2

    lists_of_indices_per_pair = [[] for _ in range(n_possible_label_pairs)]
    possible_label_pairs = []

    for i in range(n_classes):
        for j in range(n_classes):
            possible_label_pairs.append((i, j))

    # for i in range(source_y.shape[0]):
    for i in range(len(source_y)):
        src_label = source_y[i]
        tgt_label = target_y[i]
        idx = possible_label_pairs.index((src_label, tgt_label))
        lists_of_indices_per_pair[idx].append(i)

    for pair_list in lists_of_indices_per_pair:
        # pair_list = np.array(pair_list)
        n_list_trials = len(pair_list)
        shuffle = rng is not None
        folds = get_balanced_batches(n_list_trials, rng, shuffle,
                                     n_batches=n_folds)
        list_test_inds = folds[i_test_fold]
        list_all_inds = list(range(n_list_trials))
        list_train_inds = np.setdiff1d(list_all_inds, list_test_inds)
        assert np.intersect1d(list_train_inds, list_test_inds).size == 0
        assert np.array_equal(
            np.sort(np.union1d(list_train_inds, list_test_inds)),
            list_all_inds)
        # train_indices += pair_list[list_train_inds]
        # test_indices += pair_list[list_test_inds]
        train_indices += [pair_list[i] for i in list_train_inds]
        test_indices += [pair_list[i] for i in list_test_inds]

    # Because indices now sorted by label pairing, shuffle them:
    # np.random.shuffle(train_indices)
    # np.random.shuffle(test_indices)
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    train_set = select_pairs_from_paired_dataset(paired_dataset, train_indices)
    test_set = select_pairs_from_paired_dataset(paired_dataset, test_indices)
    return train_set, test_set


def select_pairs_from_paired_dataset(paired_dataset, indices):
    # back to ndarray
    return SignalAndTarget(
        {
            'source': np.array(
                [paired_dataset.X['source'][i] for i in indices]),
            'target': np.array([paired_dataset.X['target'][i] for i in indices])
        },
        {
            'source': np.array(
                [paired_dataset.y['source'][i] for i in indices]),
            'target': np.array([paired_dataset.y['target'][i] for i in indices])
        }
    )


def _check_ratios(paired_dataset):
    source_y = paired_dataset.y['source']
    target_y = paired_dataset.y['target']

    n_classes = len(np.unique(paired_dataset.y['source']))
    n_possible_label_pairs = n_classes ** 2
    counter = [x for x in range(n_possible_label_pairs)]
    possible_label_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            possible_label_pairs.append((i, j))
    for i in range(source_y.shape[0]):
        src_label = source_y[i]
        tgt_label = target_y[i]
        idx = possible_label_pairs.index((src_label, tgt_label))
        counter[idx] += 1
    for i in range(n_possible_label_pairs):
        print(possible_label_pairs[i], counter[i])
