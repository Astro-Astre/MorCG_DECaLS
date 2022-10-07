from args import *
from torch import optim
from torch import nn
import torch
from matrix import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from models.focal_loss import *
from models.Xception import *
from torch.utils.data import DataLoader
from decals_dataset import *
from preprocess.utils import *
from functools import partial
import multiprocessing

MODEL_PATH = "/data/renhaoye/MorCG/pth/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-new_no/model_6.pt"
DATA_PATH = "/data/renhaoye/MorCG/dataset/out_decals/scaled/"


def pred(i, rows, w):
    data = load_img(rows[i].split(" ")[0])
    x = torch.from_numpy(data)
    y = model(x.to("cuda:0").unsqueeze(0))
    pred = (torch.max(torch.exp(y), 1)[1]).data.cpu().numpy()
    pred2 = F.softmax(torch.Tensor(y.cpu()), dim=1).detach().numpy()[0]
    prob = pred2[np.argmax(pred2)]
    w.writelines(str(rows[i].split("\n")[0]) + " " +
                 str(pred)[1] + " " +
                 str(prob) + " " +
                 str(pred2[0]) + " " +
                 str(pred2[1]) + " " +
                 str(pred2[2]) + " " +
                 str(pred2[3]) + " " +
                 str(pred2[4]) + " " +
                 str(pred2[5]) + " " +
                 str(pred2[6]) + "\n")


if __name__ == '__main__':
    if data_config.rand_seed > 0:
        init_rand_seed(data_config.rand_seed)
    # with open("/data/renhaoye/MorCG/dataset/overlap_decals_trainvalid.txt", "r") as r:
    #     data = r.readlines()
    with open("/data/renhaoye/MorCG/dataset/overlap_train.txt", "r") as r:
        data = r.readlines()
    with open("/data/renhaoye/MorCG/dataset/overlap_valid.txt", "r") as r:
        data.extend(r.readlines())
    model = torch.load(MODEL_PATH)
    device_ids = [0, 1]
    model.to("cuda:0")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = False
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    name = "out_decals_prob.txt"
    with open("/data/renhaoye/MorCG/%s" % name, "w+") as w:
        w.writelines("loc label pred prob prob_0 prob_1 prob_2 prob_3 prob_4 prob_5 prob_6\n")
    w = open("/data/renhaoye/MorCG/%s" % name, "a")
    for i in range(len(data)):
        pred(i, data, w)
    w.close()
