import pandas as pd
import numpy as np


def make_ans_file(file_path, model):
    pred = np.argmax(model.predict((test_ds_x).astype(np.float32), verbose=False), axis=1)
    ans = pd.DataFrame({'Id': np.arange(pred.shape[0]), 'Category': pred})
    ans.to_csv(file_path, index=False)
    return pred
