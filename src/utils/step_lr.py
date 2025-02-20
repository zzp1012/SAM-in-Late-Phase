from typing import List

class StepLRforWRN:
    def __init__(self, learning_rate: float, total_epochs: int):
        """_summary_

        Args:
            learning_rate (float): _description_
            total_epochs (int): _description_
        """
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, optimizer, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        # elif epoch < self.total_epochs * 8/10:
        #     lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 2

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class MultiStepLR:
    def __init__(self, learning_rate: float, milestones: List[int], gamma: float):
        """_summary_

        Args:
            learning_rate (float): _description_
            milestones (List[int]): _description_
            gamma (float): _description_
        """
        self.milestones = milestones
        self.base = learning_rate
        self.gamma = gamma
        
    def __call__(self, optimizer, epoch):
        lr = self.base
        for milestone in self.milestones:
            if epoch >= milestone - 1:
                lr *= self.gamma

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
