"""Loss for encounter-rate targets.

A training row is one cell, week and taxon group with effort ``n`` (observer-days).
Its target for each species of the group is the encounter rate ``k / n``, a number
in [0, 1], not a 0/1 label. Binary cross-entropy against that soft target is the
binomial log-likelihood per observer-day, so ``sigmoid(logit)`` learns the rate
itself. Each row is weighted by ``n / (n + c)``: a row with plenty of effort
counts fully, a row backed by one observer-day little, and a cell with 50,000
checklists does not outweigh a thousand cells with 50. Species of other groups
are masked out (a mammal survey says nothing about birds).

The release model uses plain BCE. The other options are kept for ablations:

* ``link='cloglog'``: rate = 1 - exp(-exp(z)), the link of a Poisson abundance model.
* ``pos_weight``: scales the loss of entries with a positive rate.
* ``gamma_pos`` / ``gamma_neg``: focal down-weighting of easy entries, separately for
  positive and zero targets (``gamma_neg`` alone is the asymmetric loss without clipping).
* ``rank_weight``: adds a list-wise term, cross-entropy between ``softmax(z)`` over the
  row's species and the normalized rates, which rewards putting common species on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EncounterLoss:
    c: float = 15.0
    link: str = 'logit'
    pos_weight: float = 1.0
    gamma_pos: float = 0.0
    gamma_neg: float = 0.0
    rank_weight: float = 0.0

    def __post_init__(self):
        if self.link not in ('logit', 'cloglog'):
            raise ValueError(f'unknown link {self.link!r}')

    def probability(self, logits: torch.Tensor) -> torch.Tensor:
        if self.link == 'cloglog':
            return 1 - torch.exp(-torch.exp(logits.clamp(max=3.0)))
        return torch.sigmoid(logits)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 effort: torch.Tensor) -> torch.Tensor:
        """Weighted mean over rows of the summed per-species loss.

        Args:
            logits: ``[rows, species]`` network outputs.
            target: ``[rows, species]`` encounter rates.
            mask: ``[rows, species]``, 1 for the species of the row's group.
            effort: ``[rows]`` observer-days behind each row.
        """
        if self.link == 'cloglog':
            p = self.probability(logits).clamp(1e-7, 1 - 1e-7)
            loss = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        positive = target > 0
        if self.pos_weight != 1.0:
            loss = torch.where(positive, loss * self.pos_weight, loss)
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            p = self.probability(logits).detach()
            # Rescaled so that an entry at p = 0.5 keeps its weight.
            loss = torch.where(positive, loss * (2 * (1 - p)) ** self.gamma_pos, loss * (2 * p) ** self.gamma_neg)
        per_row = (loss * mask).sum(1)
        if self.rank_weight > 0:
            total = (target * mask).sum(1, keepdim=True)
            share = target * mask / total.clamp_min(1e-6)
            masked_logits = logits.masked_fill(mask == 0, float('-inf'))
            listwise = -(share * F.log_softmax(masked_logits, 1).nan_to_num(neginf=0.0)).sum(1)
            per_row = per_row + self.rank_weight * total.squeeze(1) * listwise
        weight = effort / (effort + self.c)
        return (weight * per_row).sum() / weight.sum().clamp_min(1e-8)
