from models import losses, efficientnet_standard
from architecture import resnet_torchvision_custom
from utils.galaxy_dataset import *
from config import *
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import schemas, label_metadata
from tqdm import tqdm


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def select_base_architecture_func_from_name(base_architecture):
    if base_architecture == 'efficientnet':
        get_architecture = efficientnet_standard.efficientnet_b0
        representation_dim = 1280
    elif base_architecture == 'resnet_detectron':
        from architecture import resnet_detectron2_custom
        get_architecture = resnet_detectron2_custom.get_resnet
        representation_dim = 2048
    elif base_architecture == 'resnet_torchvision':
        get_architecture = resnet_torchvision_custom.get_resnet  # only supports color
        representation_dim = 2048
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
        mkdir(self.config.save_dir + "log/")
        writer = torch.utils.tensorboard.SummaryWriter(self.config.save_dir + "log/")
        writer.add_graph(model.module, torch.rand(1, 3, 256, 256).cuda())
        info = transfer_config()
        with open(transfer_config.save_dir + "info.txt", "w") as w:
            for each in info.__dir__():
                attr_name = each
                attr_value = info.__getattribute__(each)
                w.write(str(attr_name) + ':' + str(attr_value) + "\n")
        for epoch in range(start + 1, self.config.epochs):
            train_loss = 0
            self.model.train()
            for i, (X, label) in enumerate(tqdm(train_loader)):
                label = torch.as_tensor(label, dtype=torch.long)
                X = X.cuda()
                label = label.cuda()
                out = self.model(X)  # 正向传播
                loss_value = torch.mean(self.loss_func(out, label))  # 求损失值, out is concentration
                self.optimizer.zero_grad()  # 优化器梯度归零
                loss_value.backward()  # 反向转播，刷新梯度值
                self.optimizer.step()
                train_loss += loss_value
            losses = (train_loss / len(train_loader))
            writer.add_scalar('Training loss by steps', losses, epoch)
            print("epoch: ", epoch)
            print("loss: ", losses)
            # eval_loss = np.zeros(8)
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
            # mkdir('%s/checkpoint' % transfer_config.save_dir)
            torch.save(checkpoint, '%s/checkpoint/ckpt_best_%s.pth' % (self.config.save_dir, str(epoch)))
            torch.save(self.model.module, '%s/model_%d.pt' % (self.config.save_dir, epoch))


train_data = GalaxyDataset(annotations_file=transfer_config.train_file, transform=transfer_config.transfer)
train_loader = DataLoader(dataset=train_data, batch_size=transfer_config.batch_size,
                          shuffle=True, num_workers=transfer_config.WORKERS, pin_memory=True)

valid_data = GalaxyDataset(annotations_file=transfer_config.valid_file, transform=transfer_config.transfer)
valid_loader = DataLoader(dataset=valid_data, batch_size=transfer_config.batch_size,
                          shuffle=True, num_workers=transfer_config.WORKERS, pin_memory=True)

model_architecture = transfer_config.model_architecture
# get_architecture, representation_dim = select_base_architecture_func_from_name(model_architecture)
MODEL_PATH = "/data/renhaoye/model_256_Adam/model_13.pt"
# model = define_model.get_plain_pytorch_zoobot_model(
#     output_dim=34,
#     include_top=True,
#     channels=3,
#     get_architecture=get_architecture,
#     representation_dim=representation_dim
# )
model = torch.load(MODEL_PATH)
model = model.cuda()
device_ids = [0, 1]
model = torch.nn.DataParallel(model, device_ids=device_ids)

optimizer = eval(transfer_config.optimizer)(model.parameters(), **transfer_config.optimizer_parm)

Trainer = trainer(model=model, optimizer=optimizer, config=transfer_config)
Trainer.train(train_loader=train_loader, valid_loader=valid_loader)
