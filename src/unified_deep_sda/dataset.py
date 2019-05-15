import os
import pickle
from itertools import combinations, combinations_with_replacement, permutations, product
import numpy as np
import time
from braindecode.datautil.splitters import concatenate_sets, select_examples
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.iterators import BalancedBatchSizeIterator, get_balanced_batches
from functools import reduce


def main():
    target_idx = 0
    bcic = BCICIV2a()
    bcic.create_siamese_dataset(target_idx, [x for x in range(9) if x != target_idx])


class BCICIV2a:
    def __init__(self, low_cut=4, seed=123456789):
        np.random.seed(seed)
        self.pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data'))
        if low_cut == 4:
            self.pickle_path = self.pickle_dir + "/bcic_iv_2a_all_9_subjects.pickle"
        else:
            self.pickle_path = self.pickle_dir + "/bcic_iv_2a_all_9_subjects_lowcut_0.pickle"

        self.data = np.array(self._get_data())
        self.n_classes = 4
        self.n_subjects = 9

    def _get_data(self):
        with open(self.pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_subject(self, subject_idx):
        """
        returns data single subject
        :param subject_idx: int in [0, ..., 8]
        :return: subject data
        """
        return self.data[subject_idx]

    def get_subjects(self, list_subject_idx):
        """
        returns data multiple subjects
        :param list_subject_idx: list of ints in [0, ..., 8]
        :return:
        """
        return self.data[list_subject_idx]

    def create_siamese_dataset(self, target_subject, source_subjects):
        target_data = self.get_subject(target_subject)  # single target subject
        source_data = self.get_subjects(source_subjects)  # multiple source subjects
        # Make pairs of target and sources suitable for siamese (two-stream) network:
        siamsese_bcic = SiameseBCICIV2A(target_data, source_data, self.n_classes, self.n_subjects)
        siamsese_bcic.create_paired_dataset()


class SiameseBCICIV2A:
    """
    Makes pairs of target and sources suitable for siamese (two-stream) network
    """

    def __init__(self, target_data, source_data, n_classes, n_subjects):
        """
        :param target_data: single target subject
        :param source_data: multiple source subjects
        :param n_classes:
        :param n_subjects:
        """
        self.target_data = target_data
        self.source_data = source_data
        self.n_classes = n_classes
        self.n_subjects = n_subjects
        if target_data is not None:
            self.n_samples_per_subject = self.target_data.y.shape[0]
            self.n_samples_per_label_per_subject = self.n_samples_per_subject // self.n_classes

        self.paired_dataset = None
        self.train_set = None
        self.validation_set = None
        self.train_set = None

    def create_paired_dataset(self, debug_check_pairs=False):
        paired_data = []

        # For each source subject get positive and negative pairs
        for subject in self.source_data:
            # Get inidices to sort data by label
            indices_sort_by_label_target = np.argsort(self.target_data.y)
            indices_sort_by_label_src_sub = np.argsort(subject.y)

            # Shuffle data points within label groups (i.e. shuffle while keeping label sort order)
            for i_label in range(self.n_classes):
                s = i_label * self.n_samples_per_label_per_subject
                e = s + self.n_samples_per_label_per_subject
                np.random.shuffle(indices_sort_by_label_target[s:e])
                np.random.shuffle(indices_sort_by_label_src_sub[s:e])

            # Create positive pair dataset (labels of pair (target and source data points) are the same)
            positive_pair_dataset = self.create_positive_pairs(subject, indices_sort_by_label_target,
                                                               indices_sort_by_label_src_sub)

            # Create negative pair dataset (pairs of src and target data points with different labels)
            negative_pair_dataset = self.create_negative_pairs(subject, indices_sort_by_label_target,
                                                               indices_sort_by_label_src_sub)

            if debug_check_pairs:
                self.check_paired_dataset(positive_pair_dataset, True)
                self.check_paired_dataset(negative_pair_dataset, False)

            paired_data.append(concatenate_sets([positive_pair_dataset, negative_pair_dataset]))

        # concatenate the paired target+src_subject to form new 8 * 2 * 576 long dataset:
        # 8 *  [num total subjects - single target_subject]
        # 2 *  [each sample (per src_sub-target pairing) once in pos and once in neg pair-dataset]
        # 576  [n_samples per subject]
        # return concatenate_sets(paired_data)
        self.paired_dataset = concatenate_sets(paired_data)

    def create_positive_pairs(self, subject, indices_target, indices_source):
        return SignalAndTarget(
            # X:
            np.array(list(zip(
                self.target_data.X[indices_target],
                subject.X[indices_source]
            ))),
            # y:
            # subject.y[indices_sort_by_label_src_sub]
            np.array(list(zip(
                self.target_data.y[indices_target],
                subject.y[indices_source]
            )))
        )

    def create_negative_pairs(self, subject, indices_target, indices_source):
        indices_source = self.create_negative_pair_source_indices(indices_source)
        return SignalAndTarget(
            # X:
            np.array(list(zip(
                self.target_data.X[indices_target],
                subject.X[indices_source]
            ))),
            # y:
            # subject.y[indices_sort_by_label_src_sub]
            np.array(list(zip(
                self.target_data.y[indices_target],
                subject.y[indices_source]
            )))
        )

    def create_negative_pair_source_indices(self, indices_src_subject):
        """
        Shifts the src subject indices around such that when zipped
        source and target data points have different labels and such that
        each type of negative pair is equally represented
        :param indices_src_subject:
        :return:
        """
        n_samples_per_permutation = self.n_samples_per_subject // len(list(permutations(range(self.n_classes), 2)))
        new_indices_sort_by_label_src_sub = []
        for i in range(self.n_classes):
            for idx, j in enumerate([x for x in range(self.n_classes) if x != i]):
                mul = [x for x in range(self.n_classes) if x != j].index(i)
                s = (j * self.n_samples_per_label_per_subject) + (mul * n_samples_per_permutation)
                e = s + n_samples_per_permutation
                new_indices_sort_by_label_src_sub.append(indices_src_subject[s:e])
        return np.concatenate(new_indices_sort_by_label_src_sub)

    def split_into_train_valid_test(self, n_folds, i_test_fold, debug=False):
        """
        :param debug: for debugging
        :param n_folds: Number of folds to split dataset into.
        :param i_test_fold: Index of the test fold (0-based). Validation fold will be immediately preceding fold.
        :return:
        """
        test_inds = np.zeros(int((1 / n_folds) * len(self.paired_dataset.y)))
        valid_inds = np.zeros(int((1 / n_folds) * len(self.paired_dataset.y)))
        train_inds = np.zeros(int(((n_folds - 2) / n_folds) * len(self.paired_dataset.y)))

        ind = np.lexsort((self.paired_dataset.y[:, 0], self.paired_dataset.y[:, 1]))
        _, counts = np.unique(self.paired_dataset.y, axis=0, return_counts=True)

        for i, count in enumerate(counts):
            s = sum(counts[:i])
            indices_slice = ind[s:s + count]

            i_valid_fold = i_test_fold - 1
            if i_test_fold == 0:
                i_valid_fold = n_folds - 1
            folds = np.split(indices_slice, n_folds)
            test_fold = folds[i_test_fold]
            valid_fold = folds[i_valid_fold]
            train_fold = np.concatenate([folds[x] for x in range(n_folds) if x not in [i_test_fold, i_valid_fold]])

            s_test_valid = int(s * (1 / n_folds))
            e_test_valid = int(s_test_valid + len(test_fold))
            test_inds[s_test_valid: e_test_valid] = test_fold
            valid_inds[s_test_valid: e_test_valid] = valid_fold

            s_train = int(s * ((n_folds - 2) / n_folds))
            e_train = s_train + len(train_fold)
            train_inds[s_train: e_train] = train_fold

            if debug:
                print(f"indices from {s} to {s + count} (count={count})")
                self.check_train_valid_test_unique_indices(test_fold, valid_fold, train_fold)
                print(f"test/valid put in from {s_test_valid} to {e_test_valid}"
                      f"\ttestfoldlen {len(test_fold)} = validfoldlen {len(valid_fold)}")
                print(f"train put in from {s_train} to {e_train}\n")

        test_set = select_examples(self.paired_dataset, test_inds.astype(int))
        valid_set = select_examples(self.paired_dataset, valid_inds.astype(int))
        train_set = select_examples(self.paired_dataset, train_inds.astype(int))

        if debug:
            self.check_set_ratios(self.paired_dataset.y, train_set, valid_set, test_set)

        return train_set, valid_set, test_set

    @staticmethod
    def check_set_ratios(paired_dataset_y, train_set, valid_set, test_set):
        pairednzc = np.count_nonzero(paired_dataset_y[:, 0] == paired_dataset_y[:, 1])
        print('complete dataset ratio (high is lots of same class pairs)', pairednzc / paired_dataset_y.shape[0])
        nzcount = np.count_nonzero(train_set.y[:, 0] == train_set.y[:, 1])
        print('train_set ratio (high is lots of same class pairs)', nzcount / train_set.y.shape[0])
        nzcount = np.count_nonzero(valid_set.y[:, 0] == valid_set.y[:, 1])
        print('valid_set ratio (high is lots of same class pairs)', nzcount / valid_set.y.shape[0])
        nzcount = np.count_nonzero(test_set.y[:, 0] == test_set.y[:, 1])
        print('test_set ratio (high is lots of same class pairs)', nzcount / test_set.y.shape[0])

    @staticmethod
    def check_train_valid_test_unique_indices(test_inds, valid_inds, train_inds):
        final_intersect = reduce(np.intersect1d, (test_inds, valid_inds, train_inds))
        assert final_intersect.shape[0] == 0, "ERROR! Indices of train/valid/test sets are not unique"
        print(f"SUCCESS! Check unique indices train/valid/test sets.")

    @staticmethod
    def check_paired_dataset(paired_dataset, is_pos_paired_dataset):
        for y in paired_dataset.y:
            if is_pos_paired_dataset:
                assert y[0] == y[1], f"ERROR! Positive paired dataset contains pair with different labels"
            elif not is_pos_paired_dataset:
                assert y[0] != y[1], "ERROR! Negative paired dataset contains pair with same labels"
        print('Paired dataset paired up correctly.')


if __name__ == '__main__':
    main()
