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

MODEL_PATH = "/data/renhaoye/MorCG/pth/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-new_no_transfer/model_14.pt"


def pred(i, rows, w):
    data = load_img(rows[i])
    x = torch.from_numpy(data)
    y = model(x.to("cuda:0").unsqueeze(0))
    pred = (torch.max(torch.exp(y), 1)[1]).data.cpu().numpy()
    pred2 = F.softmax(torch.Tensor(y.cpu()), dim=1).detach().numpy()[0]
    prob = pred2[np.argmax(pred2)]
    w.writelines(str(rows[i].split("_")[0]) + " " +
                 str(rows[i].split("_")[1]) + " " +
                 str(rows[i]) + " " +
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

    out_decals = [
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/155.22563676202859_33.22680160420511.fits"
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/226.9245734988234_32.00931149947395.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/230.34655708160926_32.10722953266542.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/37.17768509981855_-0.8492694387926467.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/145.335197798839_32.12027822848186.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/145.4789362478228_32.94797368351723.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/161.09977756577763_32.01956566436551.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/131.798072976585_31.951465035069496.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/222.27688947514443_32.02846481098737.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/166.96598776859014_32.849414661492744.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/37.7815525377259_0.314031549347303.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/135.13437009901975_32.991156368174465.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/144.83228374510026_32.30575955413059.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/175.69409021166211_32.040066415564844.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/40.92842548427848_0.9436772599353507.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/38.99291587954734_1.2574688754781835.fits",
        "/data/renhaoye/MorCG/dataset/out_decals/scaled/158.07771389284878_33.00635983176716.fits"]
    model = torch.load(MODEL_PATH)
    device_ids = [0, 1]
    model.to("cuda:0")
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = False
    with open("/data/renhaoye/decals_2022/out_decals_prob7.txt", "w+") as w:
        w.writelines("ra dec loc label prob prob_0 prob_1 prob_2 prob_3 prob_4 prob_5 prob_6\n")
    w = open("/data/renhaoye/decals_2022/out_decals_prob7.txt", "a")
    for i in range(len(out_decals)):
        # for i in range(10):
        pred(i, out_decals, w)
    w.close()
    # index = []
    # for i in range(len(out_decals)):
    #     index.append(i)
    # p = multiprocessing.Pool(8)
    # p.map(partial(augmentation, rows=out_decals, w=w), index)
    # p.close()
    # p.join()
