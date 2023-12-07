import torch


def class_balanced_focal_ce_loss(probs, ground_truth, weights, gamma):
    log_probs = -torch.log(probs)
    focals = (1 - probs) ** gamma
    class_balance = torch.reshape(weights, (1, 1, 1, -1))
    loss = torch.sum(log_probs * focals * class_balance * ground_truth)
    num_pixels = probs.shape[0] * probs.shape[1] * probs.shape[2]
    loss_avg = loss / num_pixels
    return loss_avg

def dice_index(probs, ground_truth):
    # probs: (N, H, W, C)
    intersection = probs * ground_truth # (N, H, W, C)
    inter_mag = 2 * torch.sum(intersection, dim=(1, 2)) # (N, C)
    probs_mag = torch.sum(probs ** 2, dim=(1, 2)) # (N, C)
    ground_mag = torch.sum(ground_truth ** 2, dim=(1, 2)) # (N, C)
    dice = inter_mag / (probs_mag + ground_mag) # (N, C)
    return torch.mean(dice)

def dice_loss(probs, ground_truth):
    # probs: (N, H, W, C)
    intersection = probs * ground_truth  # (N, H, W, C)
    inter_mag = 2 * torch.sum(intersection, dim=(1, 2))  # (N, C)
    probs_mag = torch.sum(probs ** 2, dim=(1, 2))  # (N, C)
    ground_mag = torch.sum(ground_truth ** 2, dim=(1, 2))  # (N, C)
    dice = inter_mag / (probs_mag + ground_mag)  # (N, C)
    loss = torch.mean(1- dice)
    return loss