import glob


def main():
    eegnet_no_tl()


def eegnet_tl_finetune():
    """
    train model using all but the target-subject as source dataset
    finetune/retrain model using targe-subject as target dataset
    (all layers finetuned, none frozen, seemed best from preliminary results)
    :return:
    """
    path = "/Users/sebas/Downloads/slurm-results/shallow_no_tl_defaultlr"
    files = glob.glob(path + "/*")

    first_test = False
    results = []
    for file in files:
        with open(file, "r+") as f:
            for l in f.readlines():
                print(l)
                if 'test ' in l:
                    split = l.split()
                    subject_index = int(split[-2])
                    test_fold_index = int(split[-1])
                if 'test_' in l:
                    if not first_test:
                        test_acc_source = float(l.split()[-1][:-3])
                        first_test = True
                    else:
                        test_acc_target = float(l.split()[-1][:-3])
            results.append([subject_index, test_fold_index, test_acc_source, test_acc_target])

    # IMPORTANT: if sort incorrect, rest of code fucked
    results.sort()

    final = []
    inter = []
    subject_index = -1
    for idx, result in enumerate(results):
        if subject_index < 0:
            subject_index = result[0]
            inter.append(result)
        elif subject_index == result[0]:
            inter.append(result)
        elif subject_index != result[0]:
            final.append(inter)
            inter = [result]
            subject_index = result[0]
        if idx == len(results) - 1:
            final.append(inter)

    avg_source_test_accs = []
    avg_target_test_accs = []
    for subject in final:
        accs_src = 0
        accs_tgt = 0
        for i in subject:
            accs_src += i[2]
            accs_tgt += i[3]
            print(f"subject: {i[0]}\ttest_fold: {i[1]}\ttest_acc_source: {i[2]}\ttest_acc_target: {i[3]}")
        avg_src_test_acc = accs_src / len(subject)
        avg_tgt_test_acc = accs_tgt / len(subject)
        avg_source_test_accs.append(avg_src_test_acc)
        avg_target_test_accs.append(avg_tgt_test_acc)
        print(f"avg_source_test_acc_subject_{subject[0][0]}: {avg_src_test_acc}"
              f"\tavg_target_test_acc_subject_{subject[0][0]}: {avg_tgt_test_acc}")

    print(f"\noverall average source test accuracy: {sum(avg_source_test_accs) / len(avg_target_test_accs)} \
    N.B. this doesnt have a lot of meaning")
    print(f"\noverall average test accuracy: {sum(avg_target_test_accs) / len(avg_target_test_accs)}")


def eegnet_no_tl():
    """
    eegnet results, no tl
    learn model from subject (target) data only, no source data involved
    :return:
    """
    path = "/Users/sebas/Downloads/slurm-results/eegnet_SDA_first"
    files = glob.glob(path + "/*")

    results = []
    for file in files:
        with open(file, "r+") as f:
            for l in f.readlines():
                if 'test ' in l:
                    split = l.split()
                    subject_index = int(split[-2])
                    test_fold_index = int(split[-1])
                if 'test_' in l:
                    test_acc = float(l.split()[-1][:-3])
            results.append([subject_index, test_fold_index, test_acc])

    # IMPORTANT: if sort incorrect, rest of code fucked
    results.sort()

    final = []
    inter = []
    subject_index = -1
    for idx, result in enumerate(results):
        if subject_index < 0:
            subject_index = result[0]
            inter.append(result)
        elif subject_index == result[0]:
            inter.append(result)
        elif subject_index != result[0]:
            final.append(inter)
            inter = [result]
            subject_index = result[0]
        if idx == len(results) - 1:
            final.append(inter)

    avg_test_accs = []
    for subject in final:
        accs = 0
        for i in subject:
            accs += i[2]
            print(f"subject: {i[0]}\ttest_fold: {i[1]}\ttest_acc: {i[2]}")
        avg_test_acc = accs / len(subject)
        avg_test_accs.append(avg_test_acc)
        print(f"avg_test_acc_subject_{subject[0][0]}: {avg_test_acc}")

    print(f"\noverall average test accuracy: {sum(avg_test_accs) / len(avg_test_accs)}")


if __name__ == '__main__':
    main()
