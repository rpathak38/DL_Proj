import torch
import torch.nn.functional as F
from torch import nn


def one_hot(mask, num_classes=21):
    mask = F.one_hot(mask, num_classes=num_classes)
    mask = mask.permute(0, 3, 1, 2)
    return mask


class ClassBalancedFocalCELoss(nn.Module):
    def __init__(self, gamma, weights):
        super().__init__()
        self.weights = weights
        self.gamma = gamma

    def forward(self, logits, ground_truth):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        focals = (1 - probs) ** self.gamma  # NCHW
        class_balance = torch.reshape(self.weights, (1, -1, 1, 1))  # 1C11
        ground_truth = one_hot(ground_truth)  # NCHW
        loss = torch.sum(log_probs * focals * class_balance * ground_truth)
        return -loss / probs.shape[0]


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, smooth=1.):
        # logits => (NCHW)
        # targets => (NHW)

        probs = F.softmax(logits, dim=1)  # NCHW

        one_hot_targets = one_hot(targets, num_classes=logits.shape[1]).float()  # NCHW

        intersection = torch.sum(probs * one_hot_targets, dim=(2, 3))  # (N, C)
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(one_hot_targets, dim=(2, 3))  # (NC)

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1. - dice.mean()

        return dice_loss


def dice_acc(logits, mask, smooth=1.):
    # logits NCHW
    # targets NHW
    preds = torch.argmax(logits, dim=1)  # NHW
    preds_hot = one_hot(preds)  # NCHW
    mask_hot = one_hot(mask)  # NCHW
    intersection = torch.sum(torch.logical_and(preds_hot, mask_hot), dim=(2, 3))
    union = torch.sum(torch.logical_or(preds_hot, mask_hot), dim=(2, 3))
    acc = 2 * intersection / union # (N, C)
    return torch.mean(torch.nanmean(acc[:, 1:], dim=1))


def pixel_acc(logits, ground_truth):
    # probs: (N, C, H, W)
    preds = torch.argmax(logits, dim=1)  # (N, H, W)
    return torch.sum(preds == ground_truth) / torch.numel(preds)
