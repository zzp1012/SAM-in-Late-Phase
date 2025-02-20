import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.avgmeter import MetricTracker
from utils.SAM import SAM, disable_running_stats, enable_running_stats, smooth_crossentropy

def evaluation(device: torch.device,
               model: nn.Module,
               testloader: DataLoader,
               smoothing: float = 0.0,
               train: bool = False):
    """evaluate the model

    Args:
        device: GPU or CPU
        model: the model to evaluate
        testloader: the test dataset loader
        smoothing: the smoothing factor
        train: the flag to show if the model is in training
    """
    if train:
        model.train()
    else:
        model.eval()
    with torch.no_grad():
        # testset
        test_losses, test_acc = [], 0.0
        for inputs, labels in tqdm(testloader):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.to(device)
            # set the outputs
            outputs = model(inputs)
            # set the loss
            losses = smooth_crossentropy(
                outputs, labels, smoothing=smoothing
            )
            # set the predictions
            preds = outputs.max(1)[1]
            # set the loss and accuracy
            test_losses.extend(losses.cpu().detach().numpy())
            test_acc += (preds == labels).sum().item()
    # print the test loss and accuracy
    test_loss = sum(test_losses) / len(testloader.dataset)
    test_acc /= len(testloader.dataset)
    return test_loss, test_acc


def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          trainset: Dataset,
          testset: Dataset,
          iters: int,
          lr: float,
          batch_size: int,
          wd: float,
          momentum: float,
          label_smoothing: float,
          rho: float,
          seed: int) -> None:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        trainset: the train dataset
        trainset_eval: the train dataset for evaluation
        testset: the test dataset
        iters: the iteration number
        lr: the learning rate
        batch_size: the batch size
        weight_decay: the weight decay
        momentum: the momentum
        rho: the rho for SAM
        label_smoothing: the label smoothing
        seed: the seed
    """
    logger = get_logger(__name__)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## set up the basic component for training
    # put the model to GPU or CPU
    model = model.to(device)
    # set the optimizer
    base_optimizer = torch.optim.SGD
    optimizer = SAM(
        filter(lambda p: p.requires_grad, model.parameters()), base_optimizer, 
        rho=rho, lr=lr, momentum=momentum, weight_decay=wd
    )
    
    ## set up the data part
    # set the testset loader
    trainloader_eval = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # create the seeds for the first phase
    seeds = random.Random(seed).sample(range(10000000), k=iters)

    # initialize the tracker
    tracker = MetricTracker()
    for itr in range(0, iters):
        logger.info(f"#########iteration {itr}....")
        set_seed(seeds[itr])
        
        # eval on the trainset
        train_loss, train_acc = evaluation(device, model, trainloader_eval, label_smoothing, True)
        logger.info(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")

        # eval on the testset
        test_loss, test_acc = evaluation(device, model, testloader, 0.0, False)
        # print the test loss and accuracy
        logger.info(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")
        
        # update the tracker
        tracker.track({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "iter": itr,
        })

        # create the batches for train
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        # iterate the trainloader
        inputs, labels = next(iter(trainloader))
        inputs, labels = inputs.to(device), labels.to(device)

        # train the model
        model.train()
        
        # set the SAM optimizer to zero grad
        optimizer.zero_grad()

        # set the first step
        enable_running_stats(model)
        # set the outputs
        outputs = model(inputs)
        # set the loss
        loss = smooth_crossentropy(
            outputs, labels, smoothing=label_smoothing
        ).mean()
        
        # backprop the loss
        loss.backward()
        # set the first step
        optimizer.first_step(zero_grad=True)
        
        # set the second step
        disable_running_stats(model)
        smooth_crossentropy(
            model(inputs), labels, smoothing=label_smoothing
        ).mean().backward()

        # save the results
        tracker.save_to_csv(os.path.join(save_path, f"escape.csv"))


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--resume_path", default=None, type=str,
                        help="the path to resume the model.")
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--model", default="vgg16_bn", type=str,
                        help='the model name.')
    parser.add_argument('--iters', default=50, type=int,
                        help="set iteration number")
    parser.add_argument("--lr", default=0.001, type=float,
                        help="set the learning rate.")
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size")
    parser.add_argument("--wd", default=1e-4, type=float,
                        help="set the weight decay")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="set the momentum rate")
    parser.add_argument("--rho", default=2.0, type=float,
                        help="set the rho for SAM")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="set the label smoothing")
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",
                         f"iters{args.iters}",
                         f"lr{args.lr}",
                         f"bs{args.bs}",
                         f"rho{args.rho}",])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

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
    trainset, testset = prepare_dataset(args.dataset, randomize=False)

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset, args.seed)
    logger.info(f"load the model from {args.resume_path}")
    model.load_state_dict(torch.load(args.resume_path, map_location="cpu"))
    logger.info(model)

    # train the model
    logger.info("#########training model....")
    train(save_path = os.path.join(args.save_path, "train"),
          device = args.device,
          model = model,
          trainset = trainset,
          testset = testset,
          iters = args.iters,
          lr = args.lr,
          batch_size = args.bs,
          wd = args.wd,
          momentum = args.momentum,
          label_smoothing = args.label_smoothing,
          rho = args.rho,
          seed = args.seed)

if __name__ == "__main__":
    main()