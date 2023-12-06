import torch


def class_balanced_focal_loss(probs, ground_truth, weights, gamma):
    log_probs = -torch.log(probs)
    focals = (1 - probs) ** gamma
    class_balance = torch.reshape(weights, (1, 1, 1, -1))
    loss = torch.sum(log_probs * focals * class_balance * ground_truth)
    return loss
