import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch_utils import assert_no_grad


def hinge_loss(positive_predictions, negative_predictions, mask=None, average=False):
    """
    Hinge pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """
    # checked, usually we need to use a threshold as soft-margin (but this function does not have it)
    loss = torch.clamp(negative_predictions -
                       positive_predictions +
                       1.0, 0.0)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    if average:
        return loss.mean()
    else:
        return loss.sum()
