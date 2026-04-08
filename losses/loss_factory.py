import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):

        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        union = probs.sum() + target.sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)

        bce_loss = self.bce(logits, target)
        
        return 0.5 * dice_loss + 0.5 * bce_loss

def get_loss(loss_type):
    if loss_type == "dice_bce":
        return DiceBCELoss()
    else:
        raise ValueError("Only 'dice_bce' is supported in this project")