import torch.nn as nn

# import internal libs
from utils import get_logger, set_seed

def prepare_model(model_name: str,
                  dataset: str,
                  ini_seed: int = 0) -> nn.Module:
    """prepare the random initialized model according to the name.

    Args:
        model_name (str): the model name
        dataset (str): the dataset name
        ini_seed (int): the initialization seed

    Return:
        the model
    """
    # set the seed
    set_seed(ini_seed)
    logger = get_logger(__name__)
    logger.info(f"prepare the {model_name} model for dataset {dataset}")
    if dataset.startswith("cifar"):
        if dataset == "cifar10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        else:
            raise ValueError(f"{dataset} is not supported.")
            
        if model_name.startswith("vgg"):
            try:
                import model.cifar_vgg as cifar_vgg
                model = cifar_vgg.__dict__[model_name](num_classes=num_classes)
            except:
                import model.cifar_vgg_plus as cifar_vgg_plus
                model = cifar_vgg_plus.__dict__[model_name](num_classes=num_classes)
        elif model_name.startswith("ResNet") and dataset == "cifar10":
            try:
                import model.cifar_resnet as cifar_resnet
                model = cifar_resnet.__dict__[model_name]()
            except:
                import model.cifar_resnet_plus as cifar_resnet_plus
                model = cifar_resnet_plus.__dict__[model_name]()
        elif model_name.startswith("WideResNet"):
            if model_name.endswith("madry"):
                import model.cifar_wide_resnet_madry as cifar_wide_resnet_madry
                model = cifar_wide_resnet_madry.__dict__[model_name](num_classes=num_classes)
            else:
                import model.cifar_wide_resnet as cifar_wide_resnet
                model = cifar_wide_resnet.__dict__[model_name](num_classes=num_classes)
        else:
            raise ValueError(f"unknown model name: {model_name} for dataset {dataset}")    
    else:
        raise ValueError(f"{dataset} is not supported.")
    return model

