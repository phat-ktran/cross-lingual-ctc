import torch
from torch import nn


class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction="none")
        self.use_focal_loss = use_focal_loss

    def forward(self, logits, batch):
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Transpose from (B, T, C) to (T, B, C) for PyTorch CTCLoss
        log_probs = log_probs.transpose(1, 0)
        T, B, _ = log_probs.shape

        # Create prediction lengths tensor
        preds_lengths = torch.tensor([T] * B, dtype=torch.long, device=log_probs.device)

        labels = batch[1].to(torch.int32)
        label_lengths = batch[2].to(torch.int64)

        loss = self.loss_func(log_probs, labels, preds_lengths, label_lengths)

        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = 1.0 - weight
            weight = torch.square(weight)
            loss = loss * weight

        loss = loss.mean()
        return {"loss": loss}
