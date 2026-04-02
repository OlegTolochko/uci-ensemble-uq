from __future__ import annotations

import torch
from torch import nn


PROB_REGULARIZERS = ("none", "l1", "brier", "probability_distance")


class SoftTargetTrainingLoss(nn.Module):
    def __init__(
        self,
        *,
        prob_regularizer: str = "none",
        prob_regularizer_weight: float = 0.0,
        entropy_bonus_weight: float = 0.0,
    ):
        super().__init__()
        if prob_regularizer not in PROB_REGULARIZERS:
            raise ValueError(
                f"Unsupported probability regularizer: {prob_regularizer}"
            )
        self.prob_regularizer = prob_regularizer
        self.prob_regularizer_weight = prob_regularizer_weight
        self.entropy_bonus_weight = entropy_bonus_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)

        loss = -(targets * log_probs).sum(dim=1).mean()

        if self.prob_regularizer_weight > 0:
            distances = probabilities - targets
            if self.prob_regularizer == "l1":
                loss = loss + self.prob_regularizer_weight * distances.abs().sum(dim=1).mean()
            elif self.prob_regularizer == "brier":
                loss = loss + self.prob_regularizer_weight * distances.square().sum(dim=1).mean()
            elif self.prob_regularizer == "probability_distance":
                _, highest_target_prob_idx = targets.max(dim=1)
                distance_to_highest_target = distances.gather(1, highest_target_prob_idx.unsqueeze(1)).squeeze(1)
                loss = loss + self.prob_regularizer_weight * distance_to_highest_target.mean()

        if self.entropy_bonus_weight > 0:
            safe_probabilities = probabilities.clamp_min(1e-8)
            entropy = -(safe_probabilities * safe_probabilities.log()).sum(dim=1).mean()
            loss = loss - self.entropy_bonus_weight * entropy

        return loss
