import torchvision.transforms as transforms


class data_config:
    root = "/data/renhaoye/zoobot_astre_log/"
    save_dir = "/data/renhaoye/MorCG_DECaLS/pth/model_Adam_1/"
    catalog_loc = "/data/renhaoye/mw_catalog.csv"
    train_file = "/data/renhaoye/MorCG_DECaLS/dataset/mw_train.txt"
    valid_file = "/data/renhaoye/MorCG_DECaLS/dataset/mw_valid.txt"
    # train_file = "/data/renhaoye/MorCG_DECaLS/dataset/train.csv"
    # valid_file = "/data/renhaoye/MorCG_DECaLS/dataset/valid.csv"
    model_architecture = "efficientnet"
    epochs = 1000
    batch_size = 256
    accelerator = "gpu"
    gpus = 2
    nodes = 1
    patience = 8
    always_augment = False
    dropout_rate = 0.2
    mixed_precision = False
    WORKERS = 16
    rand_seed = 1926
    transfer = transforms.Compose([transforms.ToTensor()])
    lr = 0.0001  # 学习率
    # optimizer = "torch.optim.AdamW"  # 优化器方法名称，eval()调用
    # optimizer_parm = {'lr': lr, 'weight_decay': 0.01}  # 优化器参数
    optimizer = "torch.optim.Adam"  # 优化器方法名称，eval()调用
    optimizer_parm = {'lr': lr, 'betas': (0.9, 0.999)}  # 优化器参数


class transfer_config:
    root = "/data/renhaoye/zoobot_astre_log/"
    save_dir = "/data/renhaoye/model_256_Adam_transfer/"
    train_file = "/data/renhaoye/mw_overlap_train.txt"
    valid_file = "/data/renhaoye/mw_overlap_valid.txt"
    num_workers = 20
    model_architecture = "efficientnet"
    epochs = 1000
    batch_size = 128
    accelerator = "gpu"
    gpus = 2
    nodes = 1
    patience = 8
    always_augment = False
    dropout_rate = 0.2
    mixed_precision = False
    WORKERS = 12
    transfer = transforms.Compose([transforms.ToTensor()])
    lr = 0.00001  # 学习率
    # optimizer = "torch.optim.AdamW"  # 优化器方法名称，eval()调用
    # optimizer_parm = {'lr': lr, 'weight_decay': 0.01}  # 优化器参数
    optimizer = "torch.optim.Adam"  # 优化器方法名称，eval()调用
    optimizer_parm = {'lr': lr, 'betas': (0.9, 0.999), 'weight_decay': 0.1}  # 优化器参数
    # torch.optim.Adam()
