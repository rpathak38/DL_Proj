import torch
import torch.nn.functional as F


def one_hot(mask):
    mask = F.one_hot(mask, num_classes=21)
    mask = mask.permute(0, 3, 1, 2)
    return mask


def class_balanced_focal_ce_loss(logits, ground_truth, weights, gamma):
    # logits NCHW
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    focals = (1 - probs) ** gamma #NCHW
    class_balance = torch.reshape(weights, (1, -1, 1, 1)) #1C11
    ground_truth = one_hot(ground_truth) #NCHW
    loss = torch.sum(log_probs * focals * class_balance * ground_truth)
    return -loss/probs.shape[0]


def dice_index(logits, ground_truth):
    # logits: (N, C, H, W)
    probs = F.softmax(logits, dim=1)
    ground_truth = one_hot(ground_truth)  # (N, C, H, W)
    intersection = probs * ground_truth  # (N, C, H, W)
    inter_mag = 2 * torch.sum(intersection, dim=(2, 3))  # (N, C)
    preds_mag = torch.sum(probs, dim=(2, 3))  # (N, C)
    ground_mag = torch.sum(ground_truth, dim=(2, 3))  # (N, C)
    dice = inter_mag / (preds_mag + ground_mag)  # (N, C)
    return torch.nanmean(torch.nanmean(dice, dim=1))

def dice_acc(logits, mask, smooth=1.):
    # logits NCHW
    # targets NHW
    preds = torch.argmax(logits, dim=1) # NHW
    preds_hot = one_hot(preds) # NCHW
    mask_hot = one_hot(mask) # NCHW
    intersection = torch.sum(torch.logical_and(preds_hot, mask_hot), dim=(2, 3))
    union = torch.sum(torch.logical_or(preds_hot, mask_hot), dim=(2, 3))
    return intersection/union

def dice_loss(logits, targets, smooth=1.):
    # logits => (NCHW)
    # targets => (NHW)

    probs = F.softmax(logits, dim=1) # NCHW
    num_classes = logits.size(1)

    one_hot_targets = one_hot(targets, num_classes=num_classes).float() # NCHW

    intersection = torch.sum(probs * one_hot_targets, dim=(2, 3)) #(N, C)
    union = torch.sum(probs, dim=(2, 3)) + torch.sum(one_hot_targets, dim=(2, 3)) #(NC)

    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1. - dice.mean()

    return dice_loss

def pixel_acc(logits, ground_truth):
    # probs: (N, C, H, W)
    preds = torch.argmax(logits, dim=1) # (N, H, W)
    return torch.sum(preds == ground_truth) / torch.numel(preds)
