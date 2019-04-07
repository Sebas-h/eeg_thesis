import json
import pandas as pd
import jsonpickle
import numpy as np


class ExpResult:
    def __init__(self, info_json_path):
        self.info_json_path = info_json_path
        self.model = None
        self.optimzer = None
        self.df_loss_error_per_subject = []

    def run(self):
        with open(self.info_json_path) as f:
            dictionary = json.load(f)
        self.recurse_nested_dict(dictionary)

    def recurse_nested_dict(self, d):
        for key, value in d.items():
            if type(value) is dict:
                if "py/object" in value.keys() and 'DataFrame' in value['py/object']:
                    self.df_loss_error_per_subject.append(df_from_serialized_json_dict(value))
                    return
                self.recurse_nested_dict(value)
            if key == 'model':
                self.model = value
            elif key == 'optimizer':
                self.optimzer = value


def df_from_serialized_json_dict(dict_json_dataframe):
    dataframe = pd.DataFrame.from_dict(dict_json_dataframe['values'])
    dataframe.index = dataframe.index.map(int)
    dataframe = dataframe.sort_index()
    return dataframe


######################################################################

path = "/Users/sebas/code/thesis/results/sacred_runs_from_server"
experiment_num = 5

info_path = path + "/" + str(experiment_num) + "/info.json"
er = ExpResult(info_path)
er.run()
all_test_errors = []
for df in er.df_loss_error_per_subject:
    test_error = df['test_misclass'].iloc[-1]
    all_test_errors.append(test_error)
    print(
        'test acc:',
        1 - df['test_misclass'].iloc[-1], df.shape[0],
        f"max valid acc: {1 - np.min(df['valid_misclass'][:354])}",
        f"min valid loss: {np.min(df['valid_loss'][:354])}"
    )
avg_test_error = sum(all_test_errors) / len(all_test_errors)
print("Avg test acc")
print(1 - avg_test_error)

# test misclass transfer learning:
# tl_test_errors = [0.2916666667, 0.4722222222, 0.1666666667, 0.2291666667,
#                   0.5208333333, 0.4861111111, 0.1597222222, 0.2152777778, 0.2430555556]
# print('tl acc:')
# print(1 - (sum(tl_test_errors) / len(tl_test_errors)))
