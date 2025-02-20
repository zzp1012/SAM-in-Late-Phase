import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import random
import re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from pathlib import Path

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.avgmeter import MetricTracker
from utils.sharpness import H_eigval
from utils.tools import search_by_suffix

def eval_flatness(save_path: str,
                  device: torch.device,
                  model: nn.Module,
                  weights_dict: dict,
                  abridged_trainset: Dataset,
                  batch_size: int,
                  neigs: int,) -> None:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        abridged_trainset: the abridged train dataset
        batch_size: the batch size
        neigs: the number of top eigenvalues to compute
    """
    logger = get_logger(__name__)
    os.makedirs(save_path, exist_ok=True)

    # put the model to GPU or CPU
    model = model.to(device)
    model.train()
    
    # set the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    
    # initialize the tracker
    tracker = MetricTracker()
    # compute the hessian eigenvalues
    for midpoint, weights in weights_dict.items():
        model.load_state_dict(weights)
        
        eigvals = H_eigval(device=device, 
                           model=model, 
                           dataset=abridged_trainset, 
                           loss_fn=loss_fn, 
                           neigs=neigs,
                           physical_batch_size=batch_size) # Our ResNet model gives large negative eigenvalues when initialized.
        
        logger.info(f"midpoint: {midpoint}, eigval: " + " ".join([
            f"({idx}) {eigval.item()}" for idx, eigval in enumerate(eigvals)
        ]))
        
        tracker.track({
            "midpoint": midpoint,
            **{f"eigval_{idx}": eigval.item() for idx, eigval in enumerate(eigvals)}
        })
        tracker.save_to_csv(save_path / "flatness.csv")
        


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="calculate the hessian eigenvalues.")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--load_path", default=None, type=str,
                        help='the path of loading models.')
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--model", default="vgg16_bn", type=str,
                        help='the model name.')
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size")
    # settings of hessian
    parser.add_argument("--neigs", default=2, type=int,
                        help="set the number of top eigenvalues to compute")
    parser.add_argument("--abridged_size", default=5000, type=int,
                        help="set the size of abridged dataset")
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    args.save_root = Path(args.save_root)
    os.makedirs(args.save_root, exist_ok=True)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",
                         f"bs{args.bs}",])
    args.save_path = args.save_root / exp_name
    os.makedirs(args.save_path, exist_ok=True)

    return args


def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)
    # set the seed
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)
    # save the current src
    save_current_src(save_path = args.save_path)

    # show the args.
    logger.info("#########parameters settings....")
    log_settings(args)

    # prepare the dataset
    logger.info("#########preparing dataset....")
    trainset, _ = prepare_dataset(args.dataset, randomize=False)
    # take abridged dataset from trainset
    # selected_indices = random.Random(10).sample(range(len(trainset)), args.abridged_size)
    abridged_trainset = Subset(trainset, range(args.abridged_size))

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset, args.seed)
    logger.info(model)
    
    # prepare all the model weights
    weights_dict = {}
    model_paths = search_by_suffix(args.load_path, "model_final.pt")
    for model_path in model_paths:
        tag = re.findall(r"start[0-9]+_end[0-9]+", model_path)
        assert len(tag) == 1, "the tag should be unique"
        [start_point, end_point] = re.findall(r"[0-9]+", tag[0])
        
        curr_weights = torch.load(model_path, map_location="cpu")
        if "ERM_to_SAM" in model_path:
            weights_dict[int(start_point)] = curr_weights
        elif "SAM_to_ERM" in model_path:
            weights_dict[int(end_point)] = curr_weights
    
    # sort the dict
    weights_dict = dict(sorted(weights_dict.items(), key=lambda x: x[0]))
    # midpoints = [0, 40, 80, 120, 160, 200]
    # weights_dict = {midpoint: weights_dict[midpoint] for midpoint in midpoints}
        
    # train the model
    logger.info("#########train and cal hessian eigenvalues....")
    eval_flatness(save_path = args.save_path / "flatness",
                  device = args.device,
                  model = model,
                  weights_dict = weights_dict,
                  abridged_trainset = abridged_trainset,
                  batch_size = args.bs,
                  neigs = args.neigs,)


if __name__ == "__main__":
    main()
