import math
from copy import deepcopy
from scipy import linalg
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from tqdm import tqdm

def num_para(net):
    cnt = 0
    for para in net.parameters():
        cnt += para.data.numel()
    return cnt

def vecterize_grad(net):
    grad = torch.zeros(num_para(net))
    i = 0
    for p in net.parameters():
        vg = p.grad.data.view(-1)
        grad[i: i + vg.numel()] = vg.detach()
        i += vg.numel()
    return grad

def compute_eror_grad_iter(net, x, y, num_classes, np, device):
    net.zero_grad()
    net.eval()
    x = x.to(device)
    f = net(x)
    ids = torch.LongTensor(y).view(-1, 1)
    y = torch.FloatTensor(len(y), num_classes).zero_().scatter_(1, ids, 1)
    y = y.to(device)
    e = (f - y)
    num_eyes = torch.eye(num_classes)
    grad = torch.zeros(np, num_classes)
    for c in range(num_classes):
        vec = num_eyes[c,:].view(1,-1).to(device)
        e.backward(vec, retain_graph=True)
        g = vecterize_grad(net)
        grad[:, c] = g.t() 
    return grad, e.cpu().data


class AnalyzeNet:
    def __init__(self, net, dataloader_1, dataloader_2, num_samples, num_classes, device):
        net = net.to(device)
        self.device = device
        self.ns = num_samples
        self.nc = num_classes
        self.net = net
        self.dataloader_1 = dataloader_1
        self.dataloader_2 = dataloader_2
        self.np = num_para(net)
        self.errs = torch.zeros(self.ns, num_classes)
        ###
        self.gram = torch.zeros(self.ns * num_classes, self.ns * num_classes)
        self.gram_errs = torch.zeros(self.ns * num_classes, self.ns)
        ###
        self.grad_err_gram = torch.zeros(self.ns, self.ns)
        self.noise_f = 0.0
        ###
        self.gram_f = 0.0
        self.gram_tr = 0.0
        self.gram_maxeig = 0.0
        self.loss = 0.0
        self.comp = False
        
    def compute_gram_and_errs(self):
        np = self.np
        ns = self.ns
        nc = self.nc
        device = self.device
        i = 0
        for x_A, y_A in tqdm(self.dataloader_1):
            # print(i)
            grad_A, e_A = compute_eror_grad_iter(self.net, x_A, y_A, nc, np, device)
            self.errs[i, :] = e_A
            j = 0
            for x_B, y_B in self.dataloader_2:
                if i == j:
                    inner = grad_A.t() @ grad_A
                    self.gram[i*nc: (i+1)*nc, i*nc: (i+1)*nc] = inner / ns 
                    self.gram_errs[i*nc: (i+1)*nc, i: i+1] = (e_A @ inner).t() / ns
                    self.grad_err_gram[i: i+1, i: i+1] = (e_A @ inner @ (e_A.t())) / ns
                    
                if i < j:
                    grad_B, e_B = compute_eror_grad_iter(self.net, x_B, y_B, nc, np, device)
                    inner = grad_A.t() @ grad_B
                    self.gram[i*nc: (i+1)*nc, j*nc: (j+1)*nc] = inner / ns
                    self.gram[j*nc: (j+1)*nc, i*nc: (i+1)*nc] = inner.t() / ns
                    self.gram_errs[i*nc: (i+1)*nc, j: j+1] = inner @ e_B.t() / ns
                    self.gram_errs[j*nc: (j+1)*nc, i: i+1] = (e_A @ inner).t() / ns
                    gegij = (e_A @ inner @ (e_B.t())) / ns
                    self.grad_err_gram[i: i+1, j: j+1] = gegij
                    self.grad_err_gram[j: j+1, i: i+1] = gegij
                j += 1
            i += 1
        self.comp = True

    def prepare_grads(self):
        if not self.comp:
            self.compute_grads()

    def gram_fro(self):
        self.prepare_grads()
        G = self.gram
        fro = ((G * G).sum()).sqrt()
        self.gram_f = fro.item()
        return fro.item()
    
    def noise_fro(self):
        self.prepare_grads()
        ns = self.ns
        G = self.grad_err_gram
        S1, S2, S3 = 0.0, 0.0, 0.0
        
        for i in range(ns):
            for j in range(ns):
                S1 += (G[i, j].item()) ** 2
                
        for i in range(ns):
            for j in range(ns):
                for k in range(ns):
                    S2 += G[i, k].item() * G[k, j].item() / ns
                    
        for i in range(ns):
            for j in range(ns):
                S3 += G[i, j].item() / ns
        
        S = S1 - 2 * S2 + S3 ** 2
        S = np.sqrt(S)
        self.noise_f = S
        return S
    
    def gram_op(self):
        self.prepare_grads()
        G = self.gram
        G = G.numpy()
        L, V = linalg.eig(G)
        op = L[0]
        self.gram_maxeig = op
        return op.real
    
    def gram_eigs(self):
        self.prepare_grads()
        G = self.gram
        G = G.numpy()
        L, V = linalg.eig(G)
        ops = []
        for l in range(len(L)):
            ops.append(L[l].real)
        all_ops = deepcopy(ops)
        for l in range(self.np-self.ns):
            all_ops.append(0.0)
        return ops, all_ops
    
    def err_gram_eigs(self):
        self.prepare_grads()
        G = self.grad_err_gram
        G = G.numpy()
        L, V = linalg.eig(G)
        ops = []
        for l in range(len(L)):
            ops.append(L[l].real)
        all_ops = deepcopy(ops)
        for l in range(self.np-self.ns):
            all_ops.append(0.0)
        return ops, all_ops
    
    def gram_trace(self):
        self.prepare_grads()
        G = self.gram
        tr = G.trace()
        self.gram_tr = tr.item()
        return tr.item()
    
    def qua_loss(self):
        errs = self.errs
        ns = self.ns
        nc = self.nc
        self.loss = ((errs * errs).mean() * nc / 2).item()
        return self.loss

    def alignment(self):
        ns = self.ns
        nc = self.nc 
        self.prepare_grads()
        loss = self.qua_loss()
        GE = self.gram_errs
        GE2 = GE * GE
        o1 = GE2.sum()
        GEs = GE.sum(dim=1)
        o2 = (GEs * GEs).sum() / ns
        o = o1 - o2
        H_dot_F = o.item()
        H_norm = self.gram_f
        ali1 = H_dot_F / (2 * loss * H_norm ** 2)
        
        GEG_norm = self.noise_f
        ali2 = H_dot_F / (H_norm * GEG_norm)    
        ratio = o2.item() / o1.item()
        
        return ali1, ali2, ratio
    
    