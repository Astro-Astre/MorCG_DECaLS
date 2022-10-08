from models.focal_loss import *
from models.Xception import *
from astre_utils.utils import *
from tqdm import tqdm

MODEL_PATH = "/data/renhaoye/MorCG/pth/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-final/model_3.pt"


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def batch_load(img_list):
    imgs = []
    for img in range(len(img_list)):
        imgs.append(load_img(img_list[img]))
    return np.array(imgs)


def batch_pred(rows):
    T = 200
    output_list = []
    label, img_list = [], []
    for i in range(len(rows)):
        label.append(rows[i].split(" ")[1])
    for i in range(len(rows)):
        img_list.append(rows[i].split(" ")[0])
    label = np.array(label, dtype=int)
    data = batch_load(img_list)
    X = torch.from_numpy(data).cuda()
    for i in range(T):
        output_list.append(torch.unsqueeze(model(X), 0).data.cpu().numpy())
    mean = np.mean(np.array(output_list), axis=0)[0, :]  # batchsize, num_classes
    variance = np.var(np.array(output_list), axis=0)[0, :]  # batchsize, num_classes
    prob = F.softmax(torch.Tensor(mean), dim=0).numpy()  # batchsize, num_classes
    print(prob)
    pred = np.argmax(mean, axis=1)  # batchsize
    back = []
    for i in range(len(rows)):
        back.append([img_list[i], label[i], pred[i], prob[i, :], variance[i, :]])
    # back = np.array(back)
    return np.array(back)


def pred(i, rows,w):
    T = 200
    output_list = []
    img_path, label = rows[i].split(" ")
    label = int(label)
    data = load_img(img_path)
    X = torch.from_numpy(data).cuda().unsqueeze(0)
    # print(model(X), model(X).shape)
    for i in range(T):
        output_list.append(torch.unsqueeze(F.softmax(torch.Tensor(model(X)), dim=1), 0).data.cpu().numpy())
    mean = np.mean(np.array(output_list), axis=0)
    # print(mean)
    variance = np.var(np.array(output_list), axis=0)
    # prob = F.softmax(torch.Tensor(mean[0, 0, :]), dim=0).numpy()
    pred = np.argmax(mean[0, 0, :])
    # print(prob)
    # print(variance)
    w.writelines("%s %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" %(img_path, int(label),
                                                                        int(pred),
                                                                        mean[0,0,0], mean[0,0,1], mean[0,0,2], mean[0,0,3], mean[0,0,4], mean[0,0,5], mean[0,0,6],
                                                                        variance[0,0,0], variance[0,0,1], variance[0,0,2], variance[0,0,3], variance[0,0,4], variance[0,0,5], variance[0,0,6]))
    # return list((img_path, label, pred, mean, variance[0, 0, :]))


if __name__ == '__main__':
    if data_config.rand_seed > 0:
        init_rand_seed(data_config.rand_seed)
    model = torch.load(MODEL_PATH)
    device_ids = [0, 1]
    model.to("cuda:0")
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    enable_dropout(model)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = False
    with open("/data/renhaoye/MorCG/dataset/test.txt", "r") as r:
        data = r.readlines()
    # data = data[:1]
    '''batch_predict'''
    # batch_size = 1
    # num = len(data) // batch_size
    # arr = []
    # for iter in tqdm(range(num)):
    #     arr.append(batch_pred(data[iter * batch_size:(iter + 1) * batch_size]))
    # # arr.append(batch_pred(data[num * batch_size:-1]))
    # pd.DataFrame(np.array(arr).reshape((len(data), 5)), columns=["loc", "label", "pred", "prob", "var" ]).to_csv("/data/renhaoye/testset.csv")


    # batch_pred(data[:2])

    '''single_predict'''
    # arr = []
    with open("/data/renhaoye/test.txt", "w") as w:
        w.writelines("loc label pred prob_0 prob_1 prob_2 prob_3 prob_4 prob_5 prob_6 var_0 var_1 var_2 var_3 var_4 var_5 var_6\n")

    for i in tqdm(range(len(data))):
    # for i in range(1):
    #     arr.append(pred(i, data))
        w = open("/data/renhaoye/test.txt", "a")
        pred(i, data, w)
        w.close()
        # arr = np.array(arr)
        # pd.DataFrame(arr, columns=["loc", "label", "pred", "prob", "var" ]).to_csv("/data/renhaoye/test.csv")


# import numpy as np
# import pandas as pd
#
# from args import *
# from torch import optim
# from torch import nn
# import torch
# from matrix import *
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn import functional as F
# from models.focal_loss import *
# from models.Xception import *
# from torch.utils.data import DataLoader
# from decals_dataset import *
# from preprocess.utils import *
# from functools import partial
# import multiprocessing
# from tqdm import tqdm
#
# MODEL_PATH = "/data/renhaoye/MorCG/pth/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-final/model_3.pt"
#
#
# def enable_dropout(model):
#     """ Function to enable the dropout layers during test-time """
#     for m in model.modules():
#         if m.__class__.__name__.startswith('Dropout'):
#             m.train()
#
#
# def batch_load(img_list):
#     imgs = []
#     for img in range(len(img_list)):
#         imgs.append(load_img(img_list[img]))
#     return np.array(imgs)
#
#
# def batch_pred(rows):
#     T = 200
#     output_list = []
#     label, img_list = [], []
#     for i in range(len(rows)):
#         label.append(rows[i].split(" ")[1])
#     for i in range(len(rows)):
#         img_list.append(rows[i].split(" ")[0])
#     label = np.array(label, dtype=int)
#     data = batch_load(img_list)
#     X = torch.from_numpy(data).cuda()
#     for i in range(T):
#         output_list.append(torch.unsqueeze(model(X), 0).data.cpu().numpy())
#     mean = np.mean(np.array(output_list), axis=0)[0, :]  # batchsize, num_classes
#     variance = np.var(np.array(output_list), axis=0)[0, :]  # batchsize, num_classes
#     prob = F.softmax(torch.Tensor(mean), dim=0).numpy()  # batchsize, num_classes
#     pred = np.argmax(mean, axis=1)  # batchsize
#     back = []
#     for i in range(len(rows)):
#         back.append([img_list[i], label[i], pred[i], prob[i, :], variance[i, :]])
#     # back = np.array(back)
#     np.array(back).shape
#     return np.array(back)
#
#
# def pred(i, rows):
#     T = 200
#     output_list = []
#     img_path, label = rows[i].split(" ")
#     label = int(label)
#     data = load_img(img_path)
#     X = torch.from_numpy(data).cuda().unsqueeze(0)
#     for i in range(T):
#         output_list.append(torch.unsqueeze(model(X), 0).data.cpu().numpy())
#     mean = np.mean(np.array(output_list), axis=0)
#     variance = np.var(np.array(output_list), axis=0)
#     prob = F.softmax(torch.Tensor(mean[0, 0, :]), dim=0).numpy()
#     pred = np.argmax(mean[0, 0, :])
#     return list((img_path, label, pred, prob, variance[0, 0, :]))
#
#
# if __name__ == '__main__':
#     if data_config.rand_seed > 0:
#         init_rand_seed(data_config.rand_seed)
#     model = torch.load(MODEL_PATH)
#     device_ids = [0, 1]
#     model.to("cuda:0")
#     # model = torch.nn.DataParallel(model, device_ids=device_ids)
#     model.eval()
#     enable_dropout(model)
#     for param in model.parameters():
#         param.requires_grad = False
#     for param in model.fc.parameters():
#         param.requires_grad = False
#     with open("/data/renhaoye/MorCG/dataset/test.txt", "r") as r:
#         data = r.readlines()
#
#     '''batch_predict'''
#     batch_size = 2
#     num = len(data) // batch_size
#     arr = []
#     for iter in tqdm(range(num+1)):
#         # arr.append(np.zeros((batch_size, )))
#         arr.append(batch_pred(data[(iter-1) * batch_size:iter * batch_size]))
#     arr.append(batch_pred(data[(num+1) * batch_size:-1]))
#     # pd.DataFrame(np.array(arr).reshape((len(data), 5)), columns=["loc", "label", "pred", "prob", "var" ]).to_csv("/data/renhaoye/testset.csv")
#
#
#     # batch_pred(data[:2])
#
#     '''single_predict'''
#     # arr = []
#     # for i in tqdm(range(len(data))):
#     # # for i in range(1):
#     #     arr.append(pred(i, data))
#     # arr = np.array(arr)
#     # pd.DataFrame(arr, columns=["loc", "label", "pred", "prob", "var" ]).to_csv("/data/renhaoye/testset.csv")
