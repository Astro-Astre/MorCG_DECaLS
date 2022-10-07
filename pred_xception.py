import os
from args import *
from torch import optim
from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
from training import losses
from estimators import define_model
from estimators import efficientnet_standard, resnet_torchvision_custom
import random
from torch.backends import cudnn
from pytorch_galaxy_datasets.galaxy_dataset import *
import torch

MODEL_PATH = "/data/renhaoye/MorCG/pth/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-new_no/model_6.pt"


def pred(i, rows, w):
    data = load_img(rows[i].split(" ")[0])
    x = torch.from_numpy(data)
    y = model(x.to("cuda:0").unsqueeze(0))
    pred = (torch.max(torch.exp(y), 1)[1]).data.cpu().numpy()
    w.writelines(str(rows[i].split("\n")[0]) + " " + str(pred[0]) + "\n")


def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == '__main__':
    init_rand_seed(1926)
    model = torch.load(MODEL_PATH)
    device_ids = [0, 1]
    model.to("cuda:0")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    name = "overlap_test_mw.txt"
    with open("/data/renhaoye/%s" % name, "r") as r:
        x = r.readlines()
    with open("/data/renhaoye/%s_combine.txt" % name.split(".")[0], "w+") as w:
        w.writelines("loc "
                     "smooth_or_featured_smooth smooth_or_featured_featured_or_disk smooth_or_featured_artifact "
                     "disk_edge_on_yes disk_edge_on_no "
                     "has_spiral_arms_yes has_spiral_arms_no "
                     "bar_strong bar_weak bar_no "
                     "bulge_size_dominant bulge_size_large bulge_size_moderate bulge_size_small bulge_size_none "
                     "how_rounded_round how_rounded_medium how_rounded_loose "
                     "edge_on_bulge_boxy edge_on_bulge_none edge_on_bulge_rounded "
                     "spiral_winding_tight spiral_winding_medium spiral_winding_loose "
                     "spiral_arm_count_1 spiral_arm_count_2 spiral_arm_count_3 spiral_arm_count_4 "
                     "spiral_arm_count_more_than_4 spiral_arm_count_cant_tell "
                     "merging_none merging_minor_disturbance merging_major_disturbance merging_merger "

                     "smooth_or_featured_smooth_v smooth_or_featured_featured_or_disk_v smooth_or_featured_artifact_v "
                     "disk_edge_on_yes_v disk_edge_on_no_v "
                     "has_spiral_arms_yes_v has_spiral_arms_no_v "
                     "bar_strong_v bar_weak_v bar_no_v "
                     "bulge_size_dominant_v bulge_size_large_v bulge_size_moderate_v bulge_size_small_v bulge_size_none_v "
                     "how_rounded_round_v how_rounded_medium_v how_rounded_loose_v "
                     "edge_on_bulge_boxy_v edge_on_bulge_none_v edge_on_bulge_rounded_v "
                     "spiral_winding_tight_v spiral_winding_medium_v spiral_winding_loose_v "
                     "spiral_arm_count_1_v spiral_arm_count_2_v spiral_arm_count_3_v spiral_arm_count_4_v "
                     "spiral_arm_count_more_than_4_v spiral_arm_count_cant_tell_v "
                     "merging_none_v merging_minor_disturbance_v merging_major_disturbance_v merging_merger_v pred\n")
    w = open("/data/renhaoye/%s_combine.txt" % name.split(".")[0], "a")
    queue = x[1:].copy()
    for i in range(len(queue)):
        pred(i, queue, w)
    w.close()
