# -*- coding: utf-8-*-
from torch.utils.data import DataLoader
from decals_dataset import *
from astre_utils.utils import *
from training.train import *
from models.define_model import *


if data_config.rand_seed > 0:
    init_rand_seed(data_config.rand_seed)

if __name__ == '__main__':
    train_data = GalaxyDataset(annotations_file=data_config.train_file, transform=data_config.transfer)
    train_loader = DataLoader(dataset=train_data, batch_size=data_config.batch_size,
                              shuffle=True, num_workers=data_config.WORKERS, pin_memory=True, prefetch_factor=16)

    valid_data = GalaxyDataset(annotations_file=data_config.valid_file, transform=data_config.transfer)
    valid_loader = DataLoader(dataset=valid_data, batch_size=data_config.batch_size,
                              shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)

    model_architecture = data_config.model_architecture
    get_architecture, representation_dim = select_base_architecture_func_from_name(model_architecture)

    model = get_plain_pytorch_zoobot_model(
        output_dim=34,
        include_top=True,
        channels=3,
        get_architecture=get_architecture,
        representation_dim=representation_dim
    )

    device_ids = [0, 1]
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = eval(data_config.optimizer)(model.parameters(), **data_config.optimizer_parm)

    Trainer = trainer(model=model, optimizer=optimizer, config=data_config)
    Trainer.train(train_loader=train_loader, valid_loader=valid_loader)
