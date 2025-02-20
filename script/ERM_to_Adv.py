import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.attack import FastGradientSignUntargeted
from utils.step_lr import StepLRforWRN, MultiStepLR
from utils.avgmeter import MetricTracker

def robust_evaluation(device: torch.device,
                      model: nn.Module,
                      attack: FastGradientSignUntargeted,
                      dataloader: DataLoader):
    """evaluate the robustness of the model

    Args:
        device (torch.device): device
        model (nn.Module): the model to evaluate
        attack (FastGradientSignUntargeted): the attack method
        dataloader (DataLoader): the dataloader to evaluate
        
    Returns:
        std_loss, std_acc, adv_loss, adv_acc
    """
    # evaluation
    with torch.no_grad():
        # testset
        std_loss_lst, std_acc = [], 0.0
        adv_loss_lst, adv_acc = [], 0.0
        for inputs, labels in tqdm(dataloader):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.to(device)
            # set the outputs
            outputs = model(inputs, _eval=True)
            
            # update the standard loss
            std_losses = F.cross_entropy(outputs, labels, reduction='none')
            std_loss_lst.extend(std_losses.cpu().detach().numpy())
            
            # update the standard accuracy
            std_acc += (outputs.max(1)[1] == labels).sum().item()
            
            # set the adversarial inputs
            with torch.enable_grad():
                adv_inputs = attack.perturb(inputs, labels, 
                                            reduction4loss='mean', 
                                            random_start=False)
            # set the adversarial outputs
            adv_outputs = model(adv_inputs, _eval=True)
            
            # update the adversarial loss
            adv_losses = F.cross_entropy(adv_outputs, labels, reduction='none')
            adv_loss_lst.extend(adv_losses.cpu().detach().numpy())
            
            # update the adversarial accuracy
            adv_acc += (adv_outputs.max(1)[1] == labels).sum().item()
    
    # print the test loss and accuracy
    std_loss = np.mean(std_loss_lst)
    adv_loss = np.mean(adv_loss_lst)
    std_acc /= len(dataloader.dataset)
    adv_acc /= len(dataloader.dataset)
    return std_loss, std_acc, adv_loss, adv_acc
    

def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          attack: FastGradientSignUntargeted,
          trainset: Dataset,
          testset: Dataset,
          epochs: int,
          lr: float,
          batch_size: int,
          weight_decay: float,
          momentum: float,
          step_size: list,
          start_adv: int,
          end_adv: int,
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
        start_SAM: the start epoch of SAM
        end_SAM: the end epoch of SAM
        rho: the rho for SAM
        adaptive: the adaptive for SAM
        label_smoothing: the label smoothing
        step_size: the StepLR's step size
        steps_saving: the steps to save the model
        seed: the seed
    """
    logger = get_logger(__name__)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    assert 0 <= start_adv <= end_adv <= epochs, \
        f"Invalid start_adv: {start_adv}, end_SAM: {end_adv}, epochs: {epochs}"

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
    # # set the trainset loader for evaluation
    # trainloader_eval = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
    # set the testset loader 
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # create the seeds for the first phase
    seeds = random.Random(seed).sample(range(10000000), k=epochs)

    # initialize the tracker
    tracker = MetricTracker()
    for epoch in range(0, epochs):
        logger.info(f"######Epoch - {epoch}")
        set_seed(seeds[epoch])

        isAdv = start_adv <= epoch <= end_adv
        logger.info("Is AT" if isAdv else "Is ERM")
        
        # create the batches for train
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.to(device)

            if isAdv:
                adv_inputs = attack.perturb(inputs, labels, reduction4loss='mean', random_start=True)
                outputs = model(adv_inputs, _eval=False)
            else:
                outputs = model(inputs, _eval=False)
            
            # set the loss    
            loss = F.cross_entropy(outputs, labels) # reduction='mean' by default
            
            # set zero grad
            optimizer.zero_grad()
            # backprop the loss
            loss.backward()
            # update the optimizer
            optimizer.step()

        # print the train loss and accuracy
        logger.info(tracker)

        # # eval on the trainset
        # train_std_loss, train_std_acc, train_adv_loss, train_adv_acc = robust_evaluation(
        #     device, model, attack, trainloader_eval
        # )

        # eval on the testset
        test_std_loss, test_std_acc, test_adv_loss, test_adv_acc = robust_evaluation(
            device, model, attack, testloader
        )
        
        # print the test loss and accuracy
        logger.info(f"""test_std_loss: {test_std_loss}, test_std_acc: {test_std_acc};
                        test_adv_loss: {test_adv_loss}, test_adv_acc: {test_adv_acc}""")

        # update the tracker
        tracker.track({
            "test_std_loss": test_std_loss,
            "test_std_acc": test_std_acc,
            "test_adv_loss": test_adv_loss,
            "test_adv_acc": test_adv_acc,
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
    parser.add_argument("--step_size", default=[80, 160], type=int, nargs="+",
                        help="set the StepLR stepsize")
    # set the Adversarial Training setting
    parser.add_argument('--epsilon', '-e', type=float, default=0.0157, 
                        help='maximum perturbation of adversaries (4/255=0.0157)')
    parser.add_argument('--alpha', '-a', type=float, default=0.00784,
                        help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--k', '-k', type=int, default=10,
                        help='maximum iteration when generating adversarial examples')
    parser.add_argument("--perturbation_type", type=str, default="linf", choices=["linf", "l2"],
                        help='the type of the perturbation (linf or l2)')
    parser.add_argument("--start_adv", default=150, type=int,
                        help="set the start epoch of Adversarial Training")
    parser.add_argument("--end_adv", default=160, type=int,
                        help="set the end epoch of Adversarial Training")
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
                         f"bs{args.bs}",
                         f"epsilon{args.epsilon}",
                         f"alpha{args.alpha}",
                         args.perturbation_type,])
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
    if args.dataset.startswith("cifar") and args.model.startswith("WideResNet"):
        trainset, testset = prepare_dataset(args.dataset, normalize=False, cutout=False)
    else:
        trainset, testset = prepare_dataset(args.dataset, normalize=False)

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset, args.seed)
    logger.info(model)
    
    # prepare the attack
    attack = FastGradientSignUntargeted(
        device = args.device,
        model = model,
        epsilon = args.epsilon,
        alpha = args.alpha,
        min_val = 0,
        max_val = 1,
        max_iters = args.k,
        _type = args.perturbation_type
    )

    # train the model
    logger.info("#########training model....")
    train(save_path = os.path.join(args.save_path, "train"),
          device = args.device,
          model = model,
          attack = attack,
          trainset = trainset,
          testset = testset,
          epochs = args.epochs,
          lr = args.lr,
          batch_size = args.bs,
          weight_decay = args.wd,
          momentum = args.momentum,
          step_size = args.step_size,
          start_adv=args.start_adv,
          end_adv=args.end_adv,
          seed = args.seed)

if __name__ == "__main__":
    main()
