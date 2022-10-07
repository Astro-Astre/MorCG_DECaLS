# -*- coding: utf-8-*-

from args import *
from torch import optim
from torch import nn
from tqdm import tqdm
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
from grad_cam_utils import *
from models.data_parallel import *
import pickle as pkl

if data_config.rand_seed > 0:
    init_rand_seed(data_config.rand_seed)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def mk_all_dir(config):
    mkdir(config.model_path)
    mkdir(config.model_path + "log/")
    mkdir(config.model_path + "cfm/")
    mkdir('%s/checkpoint' % config.model_path)


def save_config():
    info = data_config()
    with open(data_config.model_path + "info.txt", "w") as w:
        for each in info.__dir__():
            attr_name = each
            attr_value = info.__getattribute__(each)
            w.write(str(attr_name) + ':' + str(attr_value) + "\n")



class Trainer:
    def __init__(self, loss_func, model, optimizer, config):
        self.loss_func = loss_func
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def train(self, train_loader, valid_loader):
        print("你又来炼丹辣！")
        train_losses, train_acces = [], []
        start = -1
        mk_all_dir(self.config)
        writer = torch.utils.tensorboard.SummaryWriter(self.config.model_path + "log/")
        writer.add_graph(model.module, torch.rand(1, 3, 256, 256).cuda())
        if data_config.resume:  # contin = True continue training
            path_checkpoint = '%s/checkpoint/ckpt_best_%d.pth' % (
                data_config.model_path, data_config.last_epoch)  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点
            model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start = checkpoint['epoch']  # 读取上次的epoch
            print('start epoch: ', data_config.last_epoch + 1)

        for epoch in range(start + 1, self.config.epochs):
            print("epoch: ", epoch)
            train_loss, train_acc = 0, 0
            self.model.train()
            for i, (X, label) in enumerate(tqdm(train_loader)):
                label = torch.as_tensor(label, dtype=torch.long)
                # '''******** - 非分布式 -********'''
                # # X, label = X.to(self.config.device), label.to(self.config.device)
                '''******** - 分布式 -********'''
                X, label = X.cuda(), label.cuda()
                out = self.model(X)
                loss_value = self.loss_func(out, label)
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()
                train_loss += float(loss_value)
                _, pred = out.max(1)  # get the predict label
                num_correct = (pred == label).sum()  # compute the sum of correct pred
                acc = int(num_correct) / X.shape[0]  # compute the precision
                train_acc += acc  # accumilate the acc to compute the
            writer.add_scalar('Training loss by steps', train_loss / len(train_loader), epoch)
            writer.add_scalar('Training accuracy by steps', train_acc / len(train_loader), epoch)
            # pic, kappa = cf_metrics(train_loader, self.model, False)
            # writer.add_scalar('Training kappa by steps', kappa, epoch)
            # writer.add_figure("Confusion matrix train", pic, epoch)
            train_losses.append(train_loss / len(train_loader))
            train_acces.append(100 * train_acc / len(train_loader))
            print("train_loss: ", train_loss / len(train_loader))
            print("train_acc:", train_acc / len(train_loader))
            eval_losses, eval_acces = [], []
            eval_loss, eval_acc = 0, 0
            # T = 3
            with torch.no_grad():
                self.model.eval()
                enable_dropout(model)
                for X, label in tqdm(valid_loader):
                    output_list = []
                    label = torch.as_tensor(label, dtype=torch.long)
                    # '''******** - 非分布式 -********'''
                    # # X, label = X.to(self.config.device), label.to(self.config.device)
                    X = X.cuda()
                    label = label.cuda()
                    '''**no dropout**'''
                    test_out = self.model(X)
                    '''**mc dropout**'''
                    # for i in range(T):
                    #     output_list.append(torch.unsqueeze(self.model(X), 0))
                    # output_mean = torch.cat(output_list, 0).mean(0)
                    test_loss = self.loss_func(test_out, label)
                    # test_loss = self.loss_func(output_mean, label)
                    eval_loss += float(test_loss)
                    _, pred = test_out.max(1)
                    # _, pred = output_mean.max(1)
                    num_correct = (pred == label).sum()
                    acc = int(num_correct) / X.shape[0]
                    eval_acc += acc
            cfm, kappa = cf_m(valid_loader, self.model)
            with open(self.config.model_path + "cfm/epoch_%d_test.dat" % epoch, "wb") as w:
                pkl.dump(cfm, w)
            writer.add_figure("Confusion matrix valid", cf_map(cfm), epoch)
            eval_losses.append(eval_loss / len(valid_loader))
            eval_acces.append(eval_acc / len(valid_loader))
            writer.add_scalar('Validating loss by steps', eval_loss / len(valid_loader), epoch)
            writer.add_scalar('Validating accuracy by steps', eval_acc / len(valid_loader), epoch)
            writer.add_scalar('Validating kappa by steps', kappa, epoch)
            print("valid_loss: " + str(eval_loss / len(valid_loader)))
            print("valid_acc:" + str(eval_acc / len(valid_loader)) + '\n')
            checkpoint = { "net": self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), "epoch": epoch}
            torch.save(checkpoint, '%s/checkpoint/ckpt_best_%s.pth' % (self.config.model_path, str(epoch)))
            torch.save(self.model.module, '%s/model_%d.pt' % (self.config.model_path, epoch))


train_data = DecalsDataset(annotations_file=data_config.train_file, transform=data_config.transfer, debug=data_config.debug)
train_loader = DataLoader(dataset=train_data, batch_size=data_config.batch_size,
                          shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)

valid_data = DecalsDataset(annotations_file=data_config.valid_file, transform=data_config.transfer, debug=data_config.debug)
valid_loader = DataLoader(dataset=valid_data, batch_size=data_config.batch_size,
                          shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)

model = eval(data_config.model_name)(**data_config.model_parm)
model = model.cuda()
device_ids = [0, 1]
model = torch.nn.DataParallel(model, device_ids=device_ids)
# model = BalancedDataParallel(12, model, dim=0, device_ids=[0, 1])

loss_func = eval(data_config.loss_func)(**data_config.loss_func_parm)
optimizer = eval(data_config.optimizer)(model.parameters(), **data_config.optimizer_parm)

trainer = Trainer(loss_func=loss_func, model=model, optimizer=optimizer, config=data_config)
trainer.train(train_loader=train_loader, valid_loader=valid_loader)
