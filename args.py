from preprocess.utils import *
import torchvision.transforms as transforms

'''
parameter setting
'''


class data_config:
    input_channel = 3  # 输入通道数量，如果需要加mask，要改这里
    num_class = 7  # 类别储量，还要对应改weight，classes
    resume = False  # 断点续训
    last_epoch = 6  # 从哪个epoch开始训练
    model_name = "x_ception"  # 模型方法名，eval()调用，
    # model_name = "swin"  # 模型方法名，eval()调用，
    # model_parm = {'input_channels': input_channel, 'num_class': num_class}
    model_parm = {'dropout': 0.4}  # 模型参数
    '''***********- dataset and directory -*************'''
    root_path = '/data/renhaoye/MorCG/'  # 根目录
    origin = "decals"
    name = "BEST"
    train_file = '/data/renhaoye/MorCG/dataset/train.txt'  # 训练集txt文件
    valid_file = '/data/renhaoye/MorCG/dataset/valid.txt'  # 验证集txt文件
    test_file = '/data/renhaoye/MorCG/dataset/test.txt'  # 测试集txt文件
    transfer = transforms.Compose([transforms.ToTensor()])

    '''***********- Hyper Arguments -*************'''
    WORKERS = 12  # dataloader进程数量
    epochs = 30  # 训练总epoch
    batch_size = 32  # 批处理大小
    gamma = 2  # focal_loss超参数
    rand_seed = 1926  # 随机种子
    lr = 0.0001  # 学习率
    momentum = 0.9  # SGD的动量设置
    # weight = get_weight([2174, 17480, 17024, 9937, 6902, 6087, 1291])
    weight = get_weight([2119, 17045, 16572, 9674, 6745, 5935, 1261])
    # optimizer = "torch.optim.Adam"
    optimizer = "torch.optim.AdamW"  # 优化器方法名称，eval()调用
    optimizer_parm = {'lr': lr, 'weight_decay': 0.03}  # 优化器参数
    # optimizer = "torch.optim.Adam"  # 优化器方法名称，eval()调用
    # optimizer_parm = {'lr': lr}  # 优化器参数
    # loss_func = 'torch.nn.CrossEntropyLoss'  # 损失函数方法名称，eval()调用
    # loss_func_parm = {}  # 损失函数参数
    loss_func = 'focal_loss'
    loss_func_parm = {'alpha': weight, 'gamma': gamma, 'num_classes': num_class}
    device = "cuda:0"  # gpu
    # local_rank = 0, 1
    multi_gpu = False  # 多卡设置
    other = "final"  # 模型保存文件夹备注
    model_path = root_path + 'pth/%s-LR_%s-LS_%s-CLS_%s-BSZ_%s-OPT_%s-%s/' \
                 % (model_name, str(lr), loss_func, str(num_class), str(batch_size), optimizer.split(".")[-1], other)
    metrix_save_path = "/data/renhaoye/MorCG/sdss_pred_newFL.jpg"
    classes = (
        "merger", "round", "between", "cigar",
        "edgeOn", "noBar", "strongBar")  # 类别名称
    debug = False
