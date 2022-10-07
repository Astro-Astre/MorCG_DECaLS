# import os
# from args import *
# import pandas as pd
#
# from shared import label_metadata, schemas
# from training import train
#
#
# if __name__ == '__main__':
#     question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs
#     dependencies = label_metadata.decals_ortho_dependencies
#     schema = schemas.Schema(question_answer_pairs, dependencies)
#     catalog = pd.concat(
#         map(pd.read_csv, data_config.catalog_loc),
#         ignore_index=True)
#
#     train.train_zoobot(
#         save_dir=data_config.save_dir,
#         catalog=catalog,
#         schema=schema,
#         model_architecture=data_config.model_architecture,
#         batch_size=data_config.batch_size,
#         epochs=data_config.epochs,
#         patience=data_config.patience,
#         accelerator=data_config.accelerator,
#         nodes=data_config.nodes,
#         gpus=data_config.gpus,
#         num_workers=data_config.num_workers,
#         mixed_precision=data_config.mixed_precision,
#     )
