import torch
import torch.nn.functional as F

def one_hot(mask):
    mask = F.one_hot(mask, num_classes=21)
    mask = mask.permute(0, 3, 1, 2)
    return mask
def class_balanced_focal_ce_loss(logits, ground_truth, weights, gamma):
    # probs NCHW
    probs=F.softmax(logits, dim=1)
    log_probs = -torch.log(probs)
    focals = (1 - probs) ** gamma
    class_balance = torch.reshape(weights, (1, -1, 1, 1))
    ground_truth = one_hot(ground_truth)
    loss = torch.sum(log_probs * focals * class_balance * ground_truth)
    num_pixels = probs.shape[0] * probs.shape[2] * probs.shape[3]
    loss_avg = loss / num_pixels
    return loss_avg


# def dice_index(probs, ground_truth):
#     # probs: (N, C, H, W)
#     ground_truth = one_hot(ground_truth) # (N, C, H, W)
#     preds = probs > 0.5
#     intersection = preds * ground_truth  # (N, C, H, W)
#     inter_mag = 2 * torch.sum(intersection, dim=(2, 3))  # (N, C)
#     preds_mag = torch.sum(preds, dim=(2, 3))  # (N, C)
#     ground_mag = torch.sum(ground_truth, dim=(2, 3))  # (N, C)
#     dice = inter_mag / (preds_mag + ground_mag)  # (N, C)
#     return torch.mean(dice)


def dice_index(outputs, labels):
    intersection = torch.sum(torch.logical_and(labels, outputs))
    union = torch.sum(torch.logical_or(labels, outputs))
    dice_coefficient = (2.0 * intersection) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
    return dice_coefficient


def calculate_iou(outputs, labels):
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou.item()

def pixel_acc(preds, ground_truth):
    return torch.sum(preds == ground_truth) / torch.numel(preds)

# def dice_loss(probs, ground_truth):
#     # probs: (N, H, W, C)
#     ground_truth = one_hot(ground_truth) # (N, C, H, W)
#     intersection = probs * ground_truth  # (N, H, W, C)
#     inter_mag = 2 * torch.sum(intersection, dim=(2, 3))  # (N, C)
#     probs_mag = torch.sum(probs ** 2, dim=(2, 3))  # (N, C)
#     ground_mag = torch.sum(ground_truth ** 2, dim=(2, 3))  # (N, C)
#     dice = inter_mag / (probs_mag + ground_mag)  # (N, C)
#     loss = torch.mean(1 - dice)
#     return loss

def dice_loss(logits, targets, smooth=1.):
    probs = F.softmax(logits, dim=1)
    num_classes = logits.size(1)
    
    one_hot_targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

    intersection = torch.sum(probs * one_hot_targets, dim=(2, 3))
    union = torch.sum(probs, dim=(2, 3)) + torch.sum(one_hot_targets, dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1. - dice.mean()

    return dice_loss