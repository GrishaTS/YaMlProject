import pandas as pd
import numpy as np


def make_ans_file(file_path, pred):
    ans = pd.DataFrame({'Id': np.arange(pred.shape[0]), 'Category': pred})
    ans.to_csv(file_path, index=False)
