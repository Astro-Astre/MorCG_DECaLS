# -*- coding: utf-8-*-
from args import *
from torch import optim
from torch import nn
import torch
from matrix import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.focal_loss import *
from preprocess.utils import *
from models.Xception import *
from torch.utils.data import DataLoader
from decals_dataset import *
from preprocess.utils import *
from models.data_parallel import *
import pickle as pkl

if data_config.rand_seed > 0:
    init_rand_seed(data_config.rand_seed)


class Tester:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def test(self, test_loader):
        torch.no_grad()
        self.model.eval()
        cfm = cf_m(test_loader, self.model)
        # with open(self.config.model_path + "cfm/test.dat", "wb") as w:
        with open("/data/renhaoye/MorCG/test.dat", "wb") as w:
            pkl.dump(cfm, w)
        fig = cf_map(cfm)
        fig.savefig("/data/renhaoye/MorCG/overlap_out_decals_valid_no_transfer.jpg", dpi = 300)


test_data = DecalsDataset(annotations_file="/data/renhaoye/MorCG/dataset/overlap_valid.txt", transform=data_config.transfer)
test_loader = DataLoader(dataset=test_data, batch_size=data_config.batch_size,
                         shuffle=False, num_workers=data_config.WORKERS, pin_memory=True)

# MODEL_PATH = "/data/renhaoye/MorCG/pth/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-new/model_5.pt"
MODEL_PATH = "/data/renhaoye/MorCG/pth/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-new_no/model_6.pt"
model = torch.load(MODEL_PATH)
model = model.cuda()
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = False
device_ids = [0, 1]
model = torch.nn.DataParallel(model, device_ids=device_ids)
# model = BalancedDataParallel(12, model, dim=0, device_ids=[0, 1])

tester = Tester(model=model, config=data_config)
tester.test(test_loader=test_loader)