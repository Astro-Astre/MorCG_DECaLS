import torch
from torch import nn

from models import efficientnet_custom, custom_layers, efficientnet_standard


def get_plain_pytorch_zoobot_model(
        output_dim,
        weights_loc=None,
        include_top=True,
        channels=1,
        use_imagenet_weights=False,
        always_augment=True,
        dropout_rate=0.2,
        get_architecture=efficientnet_standard.efficientnet_b0,
        representation_dim=1280  # or 2048 for resnet
):
    """
    Create a trainable efficientnet model.
    First layers are galaxy-appropriate augmentation layers - see :meth:`utils.architecture.define_model.add_augmentation_layers`.
    Expects single channel image e.g. (300, 300, 1), likely with leading batch dimension.

    Optionally (by default) include the head (output layers) used for GZ DECaLS.
    Specifically, global average pooling followed by a dense layer suitable for predicting dirichlet parameters.
    See ``efficientnet_custom.custom_top_dirichlet``

    Args:
        output_dim (int): Dimension of head dense layer. No effect when include_top=False.
        input_size (int): Length of initial image e.g. 300 (asmeaned square)
        crop_size (int): Length to randomly crop image. See :meth:`utils.architecture.define_model.add_augmentation_layers`.
        resize_size (int): Length to resize image. See :meth:`utils.architecture.define_model.add_augmentation_layers`.
        weights_loc (str, optional): If str, load weights from efficientnet checkpoint at this location. Defaults to None.
        include_top (bool, optional): If True, include head used for GZ DECaLS: global pooling and dense layer. Defaults to True.
        expect_partial (bool, optional): If True, do not raise partial match error when loading weights (likely for optimizer state). Defaults to False.
        channels (int, default 1): Number of channels i.e. C in NHWC-dimension inputs. 

    Returns:
        tf.keras.Model: trainable efficientnet model including augmentations and optional head
    """

    modules_to_use = []

    effnet = get_architecture(
        input_channels=channels,
        use_imagenet_weights=use_imagenet_weights,
        include_top=False,  # no final three layers: pooling, dropout and dense
    )
    modules_to_use.append(effnet)

    if include_top:
        assert output_dim is not None
        # modules_to_use.append(tf.keras.layers.GlobalAveragePooling2D())  # included already in standard effnet in pytorch version - "AdaptiveAvgPool2d"
        modules_to_use.append(custom_layers.PermaDropout(dropout_rate))
        modules_to_use.append(
            efficientnet_custom.custom_top_dirichlet(representation_dim, output_dim))  # unlike tf version, not inplace

    if weights_loc:
        raise NotImplementedError
    #     load_weights(model, weights_loc, expect_partial=expect_partial)

    model = nn.Sequential(*modules_to_use)

    return model
