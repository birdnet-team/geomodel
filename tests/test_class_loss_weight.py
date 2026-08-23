"""Tests for the per-class loss-level multiplier (model/loss.py: class_loss_weight).

The multiplier up-weights an under-represented class (e.g. mammals/bats, which
GBIF records ~30-140x less than common birds) at the LOSS level — it scales the
per-species BCE, never the target value. This keeps the targets binary while
letting the optimiser pay more attention to the boosted classes.

Run directly (``python tests/test_class_loss_weight.py``) or via pytest.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.loss import AssumeNegativeLoss, focal_loss, asymmetric_loss  # noqa: E402


def _fixed_batch(n_species=4):
    """One sample, a positive on species 2, deterministic logits."""
    torch.manual_seed(0)
    logits = torch.tensor([[0.2, -0.5, 0.3, -1.0]])
    targets = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    return logits, targets


def test_assume_negative_weight_scales_loss():
    """Boosting one species raises the loss vs the unweighted baseline."""
    logits, targets = _fixed_batch()
    base = AssumeNegativeLoss()(logits, targets)

    weight = torch.tensor([1.0, 1.0, 10.0, 1.0])  # boost species 2 (the positive)
    boosted = AssumeNegativeLoss(class_loss_weight=weight)(logits, targets)

    assert boosted > base, f"weighted loss {boosted} should exceed base {base}"


def test_unit_weight_is_a_noop():
    """An all-ones multiplier must not change the loss."""
    logits, targets = _fixed_batch()
    base = AssumeNegativeLoss()(logits, targets)
    ones = AssumeNegativeLoss(class_loss_weight=torch.ones(4))(logits, targets)
    assert torch.allclose(base, ones, atol=1e-6), f"{base} vs {ones}"


def test_focal_and_asl_accept_weight():
    """focal_loss / asymmetric_loss apply the per-species weight before reduction."""
    logits, targets = _fixed_batch()
    w = torch.tensor([1.0, 1.0, 10.0, 1.0])
    assert focal_loss(logits, targets, weight=w) > focal_loss(logits, targets)
    assert asymmetric_loss(logits, targets, weight=w) > asymmetric_loss(logits, targets)


def test_weight_targets_unchanged():
    """Sanity: the multiplier is loss-level — it never mutates the targets.

    (Writing weights into the target VALUE collapsed bats in earlier experiments;
    this test documents that the multiplier path leaves targets binary.)
    """
    logits, targets = _fixed_batch()
    before = targets.clone()
    AssumeNegativeLoss(class_loss_weight=torch.tensor([1.0, 1.0, 6.0, 1.0]))(logits, targets)
    assert torch.equal(targets, before), "targets must be untouched by class weighting"


if __name__ == '__main__':
    test_assume_negative_weight_scales_loss()
    test_unit_weight_is_a_noop()
    test_focal_and_asl_accept_weight()
    test_weight_targets_unchanged()
    print("All class_loss_weight tests passed.")
