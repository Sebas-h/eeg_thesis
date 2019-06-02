import glob


def main():
    # siamese_cv_subject_finetune()
    siamese_multi_source()


def siamese_multi_source():
    file = "/Users/sebas/Downloads/slurm-results/eegnet_siamese_tgt_cls/MYJOB_OUTPUT.2368766"
    accs = []
    with open(file, "r+") as f:
        for l in f.readlines():
            if "Ordered" in l:
                test_acc = float(l.split()[-1][:-3])
                accs.append(test_acc)
                print(test_acc)
    print("avg test acc =", sum(accs) / len(accs))


def siamese_cv_subject_finetune():
    path = "/Users/sebas/Downloads/slurm-results/shallow_siamese_freeze_conv1"

    files = glob.glob(path + "/*")

    results = []

    for file in files:
        subject_idx = int(file.split('.')[-2])
        subject_accs = []
        with open(file, "r+") as f:
            for l in f.readlines():
                if "Ordered" in l:
                    test_acc = float(l.split()[-1][:-3])
                    subject_accs.append(test_acc)

        for idx, (test_acc_no_ft, test_acc_after_finetune) in enumerate(zip(subject_accs[0::2], subject_accs[1::2])):
            results.append([subject_idx, idx, test_acc_no_ft, test_acc_after_finetune])

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
            print(f"subject: {i[0]}\ttest_fold: {i[1]}\ttest_acc_no_ft: {i[2]}\ttest_acc_after_finetune: {i[3]}")
        avg_src_test_acc = accs_src / len(subject)
        avg_tgt_test_acc = accs_tgt / len(subject)
        avg_source_test_accs.append(avg_src_test_acc)
        avg_target_test_accs.append(avg_tgt_test_acc)
        print(f"avg_source_test_acc_subject_{subject[0][0]}: {avg_src_test_acc}"
              f"\tavg_target_test_acc_subject_{subject[0][0]}: {avg_tgt_test_acc}")

    print(f"\noverall average test accuracy without finetuning: {sum(avg_source_test_accs) / len(avg_target_test_accs)} \
    N.B. this doesnt have a lot of meaning")
    print(f"\noverall average test accuracy: {sum(avg_target_test_accs) / len(avg_target_test_accs)} "
          f"\n{sum(avg_target_test_accs) / len(avg_target_test_accs):.4f}")


def eegnet_tl_finetune():
    """
    train models using all but the target-subject as source dataset
    finetune/retrain models using targe-subject as target dataset
    (all layers finetuned, none frozen, seemed best from preliminary results)
    :return:
    """
    path = "/Users/sebas/Downloads/slurm-results/eegnet_tl_finetune"
    # path = "/Users/sebas/Downloads/slurm-results/shallow_trial_tl"
    files = glob.glob(path + "/*")

    first_test = False
    results = []
    for file in files:
        with open(file, "r+") as f:
            for l in f.readlines():
                # print(l)
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
    learn models from subject (target) data only, no source data involved
    :return:
    """
    path = "/Users/sebas/Downloads/slurm-results/eegnet_sda_finetune_all"
    # path = "/Users/sebas/Downloads/slurm-results/shallow_no_tl_defaultlr"
    files = glob.glob(path + "/*")

    # uuids = []
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

                # if "UUID" in l:
                #     uuid = l.split()[-1]

            results.append([subject_index, test_fold_index, test_acc])
            # uuids.append([subject_index, test_fold_index, uuid])

    # IMPORTANT: if sort incorrect, rest of code fucked
    results.sort()
    # uuids.sort()

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
    # print(uuids)


if __name__ == '__main__':
    main()
