import numpy as np


def get_bagging_pred(models, x):
    pred_summ = np.full((len(x), 10), 0.)
    for m in models:
        pred_summ += m.predict(x, verbose=False)
    pred_summ = np.array([np.argmax(i) for i in pred_summ])
    return pred_summ


def print_bagging_ensemble_statistic(models, names, x_val, y_val):
    if len(names) < len(models):
        print("В списке names нет или недостаточно названий моделей\n")
        return
    y_pred_en = get_bagging_pred(models=models, x=x_val)
    print("ACCURACY SCORE")
    for i in range(len(models)):
        print(f"{names[i]}:", get_accuracy(np.argmax(models[i].predict(x_val, verbose=False), axis=1), y_val))

    print("\nensemble bagging:", get_accuracy(y_pred_en, y_val))


def get_accuracy(y_pred, y_true):
    return round(np.sum(y_pred == y_true) / len(y_pred), 4)
