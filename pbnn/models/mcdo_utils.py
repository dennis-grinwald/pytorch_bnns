import numpy as np

from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, Conv2d, AvgPool2d
from torch.nn import LeakyReLU
from torch.nn import Module

def sample_function(model, dataset, n_draws=1, verbose=False):
    """Draw a realization of a random function."""
    outputs = []
    for _ in tqdm.tqdm(range(n_draws), disable=not verbose):
        freeze(model)
        outputs.append(predict(model, dataset))

    unfreeze(model)
    return torch.stack(outputs, dim=0)

def sample_proba(model, dataset, n_draws=1):
    logits = sample_function(model, dataset, n_draws=n_draws)
    return F.softmax(logits, dim=-1)

def predict_proba(model, dataset, n_draws=1):
    proba = sample_proba(model, dataset, n_draws=n_draws)
    return proba.mean(dim=0)

def predict_label(model, dataset, n_draws=1):
    proba = predict_proba(model, dataset, n_draws=n_draws)
    return proba.argmax(dim=-1)

def evaluate(model, dataset, n_draws=1):
    assert isinstance(dataset, TensorDataset)
    predicted = predict_label(model, dataset, n_draws=n_draws)
    target = dataset.tensors[1].cpu().numpy()
    return confusion_matrix(target, predicted.cpu().numpy())

class FreezableWeight(Module):
    def __init__(self):
        super().__init__()
        self.unfreeze()

    def unfreeze(self):
        self.register_buffer("frozen_weight", None)

    def is_frozen(self):
        """Check if a frozen weight is available."""
        return isinstance(self.frozen_weight, torch.Tensor)

    def freeze(self, seed):
        """Sample from the distribution and freeze."""
        raise NotImplementedError()

def freeze(module, seed):
    for mod in module.modules():
        if isinstance(mod, FreezableWeight):
            mod.freeze(seed)

    return module  # return self

def unfreeze(module):
    for mod in module.modules():
        if isinstance(mod, FreezableWeight):
            mod.unfreeze()

    return module  # return self

class PenalizedWeight(Module):
    def penalty(self):
        raise NotImplementedError()

def penalties(module):
    for mod in module.modules():
        if isinstance(mod, PenalizedWeight):
            yield mod.penalty()


class DropoutLinear(Linear, FreezableWeight):
    """Linear layer with dropout on inputs."""
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__(in_features, out_features, bias=bias)

        self.p = p

    def forward(self, input):
        if self.is_frozen():
            return F.linear(input, self.frozen_weight, self.bias)

        return super().forward(F.dropout(input, self.p, True))

    def freeze(self, seed):
        # let's draw the new weight by using seeds
        torch.manual_seed(seed)

        with torch.no_grad():

            prob = torch.full_like(self.weight[:1, :], 1 - self.p)
            feature_mask = torch.bernoulli(prob) / prob
            frozen_weight = self.weight * feature_mask

        # and store it
        self.register_buffer("frozen_weight", frozen_weight)
        


class DropoutConv2d(Conv2d, FreezableWeight):
    """2d Convolutional layer with dropout on input features."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 p=0.5):

        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode)

        self.p = p

    def forward(self, input):
        """Apply feature dropout and then forward pass through the convolution."""
        if self.is_frozen():
            return F.conv2d(input, self.frozen_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return super().forward(F.dropout2d(input, self.p, True))

    def freeze(self, seed):
        """Sample the weight from the parameter distribution and freeze it."""

        # let's draw the new weight by using seeds
        torch.manual_seed(seed)

        prob = torch.full_like(self.weight[:1, :, :1, :1], 1 - self.p)
        feature_mask = torch.bernoulli(prob) / prob

        with torch.no_grad():
            frozen_weight = self.weight * feature_mask

        self.register_buffer("frozen_weight", frozen_weight)