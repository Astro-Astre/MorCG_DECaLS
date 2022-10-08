import os
from models import losses
from models import efficientnet_standard
from config import *
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import label_metadata, schemas
from tqdm import tqdm
import numpy as np


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def select_base_architecture_func_from_name(base_architecture):
    if base_architecture == 'efficientnet':
        get_architecture = efficientnet_standard.efficientnet_b0
        representation_dim = 1280
    else:
        raise ValueError(
            'Model architecture not recognised: got model={}, expected one of [efficientnet, resnet_detectron, '
            'resnet_torchvision]'.format(
                base_architecture))

    return get_architecture, representation_dim


def get_avg_loss(loss):
    avg_loss = np.zeros(loss.shape[1])
    for i in range(loss.shape[1]):
        avg_loss[i] += loss[:, i].sum()
    return avg_loss


class trainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.question_answer_pairs = label_metadata.gz2_pairs  # 问题？
        self.dependencies = label_metadata.gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)

    def loss_func(self, preds, labels):  # pytorch convention is preds, labels
        return losses.calculate_multiquestion_loss(labels, preds,
                                                   self.schema.question_index_groups)  # my and sklearn convention is labels, preds

    def train(self, train_loader, valid_loader):
        print("你又来炼丹辣！")
        start = -1
        device_ids = [0, 1]
        mkdir(self.config.save_dir)
        mkdir(self.config.save_dir + "log/")
        writer = torch.utils.tensorboard.SummaryWriter(self.config.save_dir + "log/")
        writer.add_graph(self.model.module, torch.rand(1, 3, 256, 256).cuda())
        info = data_config()
        with open(data_config.save_dir + "info.txt", "w") as w:
            for each in info.__dir__():
                attr_name = each
                attr_value = info.__getattribute__(each)
                w.write(str(attr_name) + ':' + str(attr_value) + "\n")
        for epoch in range(start + 1, self.config.epochs):
            print("epoch: ", epoch)
            train_loss = 0
            self.model.train()
            for i, (X, label) in enumerate(tqdm(train_loader)):
            # for i, (X, label) in enumerate(train_loader):
                label = torch.as_tensor(label, dtype=torch.long)
                X = X.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                out = self.model(X)  # 正向传播
                loss_value = torch.mean(self.loss_func(out, label))  # 求损失值, out is concentration
                self.optimizer.zero_grad()  # 优化器梯度归零
                loss_value.backward()  # 反向转播，刷新梯度值
                self.optimizer.step()
                train_loss += loss_value
            losses = (train_loss / len(train_loader))
            writer.add_scalar('Training loss by steps', losses, epoch)
            print("loss: ", losses)
            eval_loss = 0
            with torch.no_grad():
                self.model.eval()
                for X, label in valid_loader:
                    label = torch.as_tensor(label, dtype=torch.long)
                    X = X.cuda()
                    label = label.cuda()
                    test_out = self.model(X)
                    test_loss = torch.mean(self.loss_func(test_out, label))
                    eval_loss += test_loss
            eval_losses = (eval_loss / len(valid_loader))
            writer.add_scalar('Validating loss by steps', eval_losses, epoch)
            print("valid_loss: " + str(eval_losses))
            checkpoint = {
                "net": self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "epoch": epoch
            }
            mkdir('%s/checkpoint' % data_config.save_dir)
            torch.save(checkpoint, '%s/checkpoint/ckpt_best_%s.pth' % (self.config.save_dir, str(epoch)))
            torch.save(self.model.module, '%s/model_%d.pt' % (self.config.save_dir, epoch))
