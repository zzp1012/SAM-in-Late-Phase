import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.step_lr import StepLRforWRN, MultiStepLR
from utils.avgmeter import MetricTracker
from utils.tools import evaluation
from utils.SAM import smooth_crossentropy

def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          trainset: Dataset,
          testset: Dataset,
          epochs: int,
          lr: float,
          batch_size: int,
          weight_decay: float,
          momentum: float,
          label_smoothing: float,
          step_size: list,
          seed: int) -> None:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        trainset: the train dataset
        testset: the test dataset
        epochs: the epochs
        lr: the learning rate
        batch_size: the batch size
        weight_decay: the weight decay
        momentum: the momentum
        label_smoothing: the label smoothing
        step_size: the StepLR's step size
        seed: the seed
    """
    logger = get_logger(__name__)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## set up the basic component for training
    # put the model to GPU or CPU
    model = model.to(device)
    # set the optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=weight_decay, momentum=momentum
    )

    # set the scheduler
    if model.__class__.__name__ == "WideResNet":
        logger.info("Notice: switch to the WideResNet default scheduler.")
        scheduler = StepLRforWRN(lr, epochs)
    else:
        scheduler = MultiStepLR(lr, step_size, gamma=0.1)
    
    ## set up the data part
    # set the testset loader 
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # create the seeds for the first phase
    seeds = random.Random(seed).sample(range(10000000), k=epochs)

    # initialize the tracker
    tracker = MetricTracker()
    for epoch in range(0, epochs):
        logger.info(f"######Epoch - {epoch}")
        set_seed(seeds[epoch])
        
        # create the batches for train
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        # train the model
        model.train()
        for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.to(device)

            # set the outputs
            outputs = model(inputs)
            # set the loss
            loss = smooth_crossentropy(
                outputs, labels, smoothing=label_smoothing
            ).mean()

            # set zero grad
            optimizer.zero_grad()
            # set the loss
            loss.backward()
            # set the optimizer
            optimizer.step()

            # set the loss and accuracy
            tracker.update({
                "train_loss": loss.item(),
                "train_acc": (outputs.max(1)[1] == labels).float().mean().item()
            }, n = inputs.size(0))

        # print the train loss and accuracy
        logger.info(tracker)

        # eval on the testset
        test_loss, test_acc, _ = evaluation(device, model, testloader)
        # print the test loss and accuracy
        logger.info(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

        # update the tracker
        tracker.track({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch": epoch,
        })

        # update the scheduler
        scheduler(optimizer, epoch)

        # save the results
        if epoch == epochs - 1:
            logger.info("save the final results")
            torch.save(model.state_dict(), 
                       os.path.join(save_path, f"model_final.pt"))
            tracker.save_to_csv(os.path.join(save_path, f"train.csv"))


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
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--model", default="vgg16_bn", type=str,
                        help='the model name.')
    parser.add_argument('--epochs', default=160, type=int,
                        help="set iteration number")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="set the learning rate.")
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size")
    parser.add_argument("--wd", default=1e-4, type=float,
                        help="set the weight decay")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="set the momentum rate")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="set the label smoothing")
    parser.add_argument("--step_size", default=[80, 160], type=int, nargs="+",
                        help="set the StepLR stepsize")
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
                         f"epochs{args.epochs}",
                         f"lr{args.lr}",
                         f"bs{args.bs}",])
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
    logger.info(model)

    # train the model
    logger.info("#########training model....")
    train(save_path = os.path.join(args.save_path, "train"),
          device = args.device,
          model = model,
          trainset = trainset,
          testset = testset,
          epochs = args.epochs,
          lr = args.lr,
          batch_size = args.bs,
          weight_decay = args.wd,
          momentum = args.momentum,
          label_smoothing = args.label_smoothing,
          step_size = args.step_size,
          seed = args.seed)

if __name__ == "__main__":
    main()
