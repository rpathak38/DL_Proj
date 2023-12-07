import torch
import torch.nn.functional as F

def one_hot(mask):
    mask = F.one_hot(mask, num_classes=21)
    mask = mask.permute(0, 3, 1, 2)
    return mask
def class_balanced_focal_ce_loss(probs, ground_truth, weights, gamma):
    # probs NCHW
    log_probs = -torch.log(probs)
    focals = (1 - probs) ** gamma
    class_balance = torch.reshape(weights, (1, -1, 1, 1))
    ground_truth = one_hot(ground_truth)
    loss = torch.sum(log_probs * focals * class_balance * ground_truth)
    num_pixels = probs.shape[0] * probs.shape[2] * probs.shape[3]
    loss_avg = loss / num_pixels
    return loss_avg


def dice_index(probs, ground_truth):
    # probs: (N, C, H, W)
    ground_truth = one_hot(ground_truth) # (N, C, H, W)
    preds = probs > 0.5
    intersection = preds * ground_truth  # (N, C, H, W)
    inter_mag = 2 * torch.sum(intersection, dim=(2, 3))  # (N, C)
    preds_mag = torch.sum(preds, dim=(2, 3))  # (N, C)
    ground_mag = torch.sum(ground_truth, dim=(2, 3))  # (N, C)
    dice = inter_mag / (preds_mag + ground_mag)  # (N, C)
    return torch.mean(dice)


def dice_loss(probs, ground_truth):
    # probs: (N, H, W, C)
    ground_truth = one_hot(ground_truth) # (N, C, H, W)
    intersection = probs * ground_truth  # (N, H, W, C)
    inter_mag = 2 * torch.sum(intersection, dim=(2, 3))  # (N, C)
    probs_mag = torch.sum(probs ** 2, dim=(2, 3))  # (N, C)
    ground_mag = torch.sum(ground_truth ** 2, dim=(2, 3))  # (N, C)
    dice = inter_mag / (probs_mag + ground_mag)  # (N, C)
    loss = torch.mean(1 - dice)
    return loss
