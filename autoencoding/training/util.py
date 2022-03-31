import torch


def class_name(x):
    return str(x.__class__).split(".")[-1][:-2]


def entropy(p):
    log_p = torch.log(p)
    log_p = torch.where(torch.isinf(log_p), torch.zeros_like(log_p), log_p)
    return -(p * log_p).sum(-1)
