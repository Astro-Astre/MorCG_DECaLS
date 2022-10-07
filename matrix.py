from args import *
import torch
from sklearn.metrics import cohen_kappa_score


def cf_metrics(loader, model, save: bool = False):
    y_pred = []  # save prediction
    y_true = []  # save ground truth
    device = "cuda:0"
    for inputs, labels in loader:
        inputs, label = inputs.to(data_config.device), inputs.to(data_config.device)
        output = model(inputs)  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth
    print("kappa: ", cohen_kappa_score(y_true, y_pred))
    classes = data_config.classes
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis],
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 10))
    if save:
        sns.heatmap(df_cm, annot=True, cmap="Blues").get_figure()
        plt.savefig(data_config.metrix_save_path)
    else:
        return sns.heatmap(df_cm, annot=True, cmap="Blues").get_figure()


def cf_m(loader, model):
    y_pred = []  # save prediction
    y_true = []  # save ground truth
    device = "cuda:0"
    for inputs, labels in loader:
        inputs, label = inputs.to(data_config.device), inputs.to(data_config.device)
        output = model(inputs)  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth
    kappa = cohen_kappa_score(y_true, y_pred)
    print("kappa: ", kappa)
    cf_matrix = confusion_matrix(y_true, y_pred)
    return cf_matrix, kappa


def cf_map(cf_matrix):
    classes = data_config.classes
    plt.figure(figsize=(12, 6))
    df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis],
                         index=[i for i in classes],
                         columns=[i for i in classes])
    return sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='.3f').get_figure()
