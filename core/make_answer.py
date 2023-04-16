import pandas as pd
import numpy as np


def make_ans_file(model, x, file_path=False):
    pred = np.argmax(model.predict((x).astype(np.float32), verbose=1), axis=1)
    if file_path:
        ans = pd.DataFrame({'Id': np.arange(pred.shape[0]), 'Category': pred})
        ans.to_csv(file_path, index=False)
    return pred
