import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


def get_bagging_pred(modelset, x_val):
    pred_summ = np.full((len(x_val), 10), 0.)
    for m in modelset:
        pred_summ += m[0].predict(x_val, verbose=False)
    pred_summ = np.array([np.argmax(i) for i in pred_summ])
    return pred_summ


def print_models_statistic(modelbase, y_val):
    print("SCORE")
    for key in modelbase.keys():
        print(f"{key}:", "accuracy:", accuracy_score(y_val, np.argmax(modelbase[key][1], axis=1)), "f1_score:",
              f1_score(y_val, np.argmax(modelbase[key][1], axis=1), average="macro"))


# Заранее просчитывает предикты для всех моделей, чтобы более быстро определять accuracy с разными комбинациями.
def get_ensemble_modelbase(models, names, x_val):
    if len(names) < len(models):
        print("В списке names нет или недостаточно названий моделей\n")
        return
    modelbase = {}
    for i in range(len(names)):
        modelbase[names[i]] = [models[i], np.array(models[i].predict(x_val, verbose=False))]
    return modelbase


def get_modeset(modelbase, selected_names):
    modelset = []
    for model_name in selected_names:
        modelset.append(modelbase[model_name])
    return modelset


def print_bagging_ensemble_statistic(modelset, y_val):
    pred_summ = np.full((len(y_val), 10), 0.)
    for model in modelset:
        pred_summ += model[1]
    y_pred_en = np.array([np.argmax(i) for i in pred_summ])
    print("SCORE")
    print("\nensemble accuracy:", accuracy_score(y_val, y_pred_en))
    print("\nensemble f1_score:", f1_score(y_val, y_pred_en, average="macro"))
    return y_pred_en
