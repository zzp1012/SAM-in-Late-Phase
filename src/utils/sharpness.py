from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def lanczos(device: torch.device,
            matrix_vector: Callable, 
            dim: int, 
            neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). 
    
    Args:
        device: GPU or CPU
        matrix_vector: the matrix-vector product
        dim: the dimension of the matrix
        neigs: the number of eigenvalues to compute

    Returns:
        the eigenvalues and eigenvectors
    """
    def mv(vec: np.ndarray): # vec: numpy array
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return matrix_vector(gpu_vec).detach().cpu() # which should be a torch tensor on CPU

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def compute_hvp(device: torch.device,
                model: nn.Module, 
                dataset: Dataset, 
                loss_fn: nn.Module,
                vector: torch.Tensor, 
                physical_batch_size) -> torch.Tensor:
    """Compute a Hessian-vector product.

    Args:
        device: GPU or CPU
        model: the model
        dataset: the dataset
        loss_fn: the loss function
        vector: the vector
        physical_batch_size: the physical batch size

    Returns:
        the Hessian-vector product
    """
    p = len(parameters_to_vector(model.parameters()))
    n = len(dataset)
    
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    
    dataloader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=False)
    for (X, y) in tqdm(dataloader):
        # move to GPU
        X, y = X.to(device), y.to(device)
        # compute the Hessian-vector product
        loss = loss_fn(model(X), y) / n
        grads = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, model.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp


def H_eigval(device: torch.device,
             model: nn.Module, 
             dataset: Dataset,
             loss_fn: nn.Module, 
             neigs: int = 6, 
             physical_batch_size: int = 1000) -> torch.Tensor:
    """Compute the leading Hessian eigenvalues.

    Args:
        device: GPU or CPU
        model: the model
        dataset: the dataset
        loss_fn: the loss function
        neigs: the number of eigenvalues to compute
        physical_batch_size: the physical batch size
    
    Returns:
        the eigenvalues
    """
    hvp_delta = lambda delta: compute_hvp(device, model, dataset, loss_fn,
        delta, physical_batch_size=physical_batch_size)
    nparams = len(parameters_to_vector((model.parameters())))
    evals, evecs = lanczos(device, hvp_delta, nparams, neigs=neigs)
    return evals
