# -*- coding: utf-8-*-

from args import *
from torch import nn
import torch
from models.Xception import *
from preprocess.utils import *
from models.data_parallel import *

if data_config.rand_seed > 0:
    init_rand_seed(data_config.rand_seed)


model = eval(data_config.model_name)(**data_config.model_parm)
model = model.cuda()
X = torch.zeros((1, 3, 256, 256))
# X = torch.as_tensor(X, dtype=torch.long)
label = torch.as_tensor(0, dtype=torch.float)
label = label.cuda()
X = X.cuda()
out = model(X)  # 正向传播