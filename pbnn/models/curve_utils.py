from itertools import product
import tqdm
import sys
import copy
from PIL import Image
from typing import List, Union

import numpy as np
import torch
import torchvision

from lucent.modelzoo import *
from lucent.misc.io import show
import lucent.optvis.objectives as objectives
import lucent.optvis.param as param
import lucent.optvis.render as render
import lucent.optvis.transform as transform
from lucent.misc.channel_reducer import ChannelReducer
from lucent.misc.io import show

from curvature.utils import (accuracy, setup, ram, vram, expected_calibration_error, predictive_entropy, negative_log_likelihood,
                    calibration_curve, get_eigenvectors)

def get_network_activations_mcdo(model, seed, imgs):

  activations = {}
  def get_activation_mcdo(name):
    def hook(model, input, output):
      activations[name] = output.detach().cpu().numpy()
    return hook

  handles = []
  for name, layer in model.named_modules():
    handles.append(layer.register_forward_hook(get_activation_mcdo(name)))

  sample_network(model, inv_factors, estimator, posterior_mean, seed)
  out = model(imgs)

  for handle in handles:
    handle.remove()

  return activations

def get_activation_variances(acts):
  var_dict = dict()
  for layer in net_acts['net_0']:

    layer_acts = torch.cat([net_acts[f'net_{i}'][layer] for i in range(len(seeds))])
    var_dict[layer] = torch.var(layer_acts, dim=0)
    
  return var_dict

def get_network_activations_single_model(model, img):

    activations = {}
    def get_activation_curve(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    handles = []
    for name, layer in model.named_modules():
        handles.append(layer.register_forward_hook(get_activation_curve(name)))

    img = img.reshape(1, 3, 224, 224).to(device)

    out = model(img)

    for handle in handles:
        handle.remove()

    return activations

def get_network_activations_curve(model, inv_factors, estimator, posterior_mean, seed, img, device):

  activations = {}
  def get_activation_curve(name):
    def hook(model, input, output):
      activations[name] = output.detach().cpu().numpy()
    return hook

  handles = []
  for name, layer in model.named_modules():
    handles.append(layer.register_forward_hook(get_activation_curve(name)))

  sample_curve_network(model, inv_factors, estimator, posterior_mean, seed)

  img = img.reshape(1, 3, 224, 224).to(device)

  out = model(img)

  for handle in handles:
    handle.remove()

  return activations

def get_sorted_channel_activations(net_acts):
    
    # Compute neuron activation variances
    var_dict = dict()
    for layer in net_acts['net_0']:
        layer_acts = np.concatenate([net_acts[f'net_{i}'][layer] for i in range(len(net_acts.keys()))])
        var_dict[layer] = np.var(layer_acts, axis=0)
    
    # Compute sorted channel activation variances
    list_of_channel_activations = []
    for layer_id in var_dict.keys():
        layer = var_dict[layer_id]
        layer_transformed = layer.reshape(layer.shape[0], -1)
        layer_channel_reduced = layer_transformed.mean(-1).reshape(-1)
        layer_channel_list = [(layer_id, i, val) for i, val in enumerate(layer_channel_reduced)]
        list_of_channel_activations += layer_channel_list
    
    list_of_channel_activations.sort(key=lambda x:x[2], reverse=True)
    
    return list_of_channel_activations

@torch.no_grad()
def get_layer(model, layer, X):
    hook = render.ModuleHook(getattr(model, layer))
    model(X)
    hook.close()
    return hook.features

@objectives.wrap_objective()
def dot_compare(layer, acts, batch=1):
    def inner(T):
        pred = T(layer)[batch]
        return -(pred * acts).sum(dim=0, keepdims=True).mean()

    return inner

def eval_nn(model, dataset, device=torch.device('cuda'), verbose=False):
    model.eval()

    with torch.no_grad():
        logits_list = torch.Tensor().to(device)
        labels_list = torch.LongTensor()

        dataset = tqdm.tqdm(dataset, disable=not verbose or len(dataset) == 1)
        for images, labels in dataset:
            dataset.set_postfix({'RAM': ram(), 'VRAM': vram()})

            logits = model(images.to(device, non_blocking=True))
            logits_list = torch.cat([logits_list, logits])
            labels_list = torch.cat([labels_list, labels])

        predictions = torch.nn.functional.softmax(logits_list, dim=1).cpu().numpy()
        labels = labels_list.numpy()

    if verbose:
        print(f"Accuracy: {accuracy(predictions, labels):.2f}% | ECE: {100 * expected_calibration_error(predictions, labels)[0]:.2f}%")

    return predictions, labels

def eval_curve_bnn(
             model,
             posterior_mean,
             seeds,
             dataset,
             inv_factors,
             estimator='kfac',
             stats=True,
             device=torch.device('cuda'),
             verbose=True):

    model.eval()
    mean_state = copy.deepcopy(model.state_dict())
    mean_predictions = 0
    stats_list = {"acc": [], "ece": [], "nll": [], "ent": []}

    with torch.no_grad():
        samples = tqdm.tqdm(range(seeds.shape[0]), disable=not verbose)
        for i, sample in enumerate(samples):
            samples.set_postfix({'RAM': ram(), 'VRAM': vram()})
            sample_curve_network(model, inv_factors, estimator, posterior_mean, seeds[sample])
            predictions, labels = eval_nn(model, dataset, device)

            print(f"Net {i}: Accuracy: {accuracy(predictions, labels):.2f}% | Net {i}: ECE: {100 * expected_calibration_error(predictions, labels)[0]:.2f}%")

            mean_predictions += predictions
            model.load_state_dict(mean_state)

            if stats:
                running_mean = mean_predictions / (sample + 1)
                stats_list["acc"].append(accuracy(running_mean, labels))
                stats_list["ece"].append(100 * expected_calibration_error(running_mean, labels)[0])
                stats_list["nll"].append(negative_log_likelihood(predictions, labels))
                stats_list["ent"].append(predictive_entropy(running_mean, mean=False))
        mean_predictions /= len(samples)

        if verbose:
            print(f"Accuracy: {accuracy(mean_predictions, labels):.2f}% | ECE: {100 * expected_calibration_error(mean_predictions, labels)[0]:.2f}% | Predictive entropies mean: {predictive_entropy(mean_predictions, mean=True):.2f}")

        return mean_predictions, labels, stats_list

def sample_curve_network(model: Union[torch.nn.Sequential, torch.nn.Module],
                               inv_factors: List[torch.Tensor],
                               estimator='kfac',
                               posterior_mean=None,
                               seeds=[42,42],
                               var_hp=1) -> None:
                               
    """Samples a new set of weights from the approximate weight posterior distribution and replaces the existing ones.

    Args:
        model: A (pre-trained) PyTorch model.
        inv_factors: The inverted factors (plus further quantities required for sampling)
        estimator: The FIM estimator. One of `block`, `diag`, `kfac`, `efb` or `inf`.
    """

    if posterior_mean is None:
      print("Please provide a posterior mean")
      return

    model.load_state_dict(posterior_mean)

    np.random.seed(seeds[0])
    torch.manual_seed(seeds[1])

    index = 0
    for module in model.modules():
        if module.__class__.__name__ in ['Linear', 'Conv2d']:
            weight = module.weight
            bias = module.bias
            if estimator in ["kfac", "efb"]:
                if estimator == "kfac":
                    a, b = inv_factors[index]  # a: First KFAC factor, b: second KFAC factor
                else:
                    a, b, scale = inv_factors[index]  # a, b: Eigenvectors of first and second KFAC factor

                z = torch.randn(a.size(0), b.size(0), device=a.device, dtype=a.dtype ) * var_hp 
                if estimator == "efb":
                    z *= scale.t()
                x = (a @ z @ b.t()).t() # Final transpose because PyTorch uses channels first

            elif estimator == "diag":
                var = inv_factors[index]
                x = var.new(var.size()).normal_() * var

            elif estimator.lower() in ["fisher", "full", "block", "block diagonal", "block_diagonal"]:
                var = inv_factors[index]
                x = var.new(var.shape[0]).normal_() @ var
                x = torch.cat([x[:weight.numel()].contiguous().view(*weight.shape),
                               torch.unsqueeze(x[weight.numel():], dim=1)], dim=1)

            elif estimator == "inf":
                a, b, c, d = inv_factors[index]
                x = sampler(a, b, c, d).reshape(a.shape[0], b.shape[0]).t()

            index += 1
            if bias is not None:
                bias_sample = x[:, -1].contiguous().view(*bias.shape)
                bias.data.add_(bias_sample)
                x = x[:, :-1]
            weight.data.add_(x.contiguous().view(*weight.shape))

# attribution map function
def compute_attribution_map(
    img,
    model,
    layer="mixed4d",
    cell_image_size=60,
    n_groups=6,
    n_steps=1024,
    batch_size=64,
    device="cpu"
):
    # First wee need, to normalize and resize the image
    img = torch.tensor(np.transpose(img, [2, 0, 1])).to(device)
    normalize = (
        transform.preprocess_inceptionv1()
        if model._get_name() == "InceptionV1"
        else transform.normalize()
    )
    transforms = transform.standard_transforms.copy() + [
        normalize,
        torch.nn.Upsample(size=224, mode="bilinear", align_corners=True),
    ]
    transforms_f = transform.compose(transforms)
    # shape: (1, 3, original height of img, original width of img)
    img = img.unsqueeze(0)
    # shape: (1, 3, 224, 224)
    img = transforms_f(img)

    # Here we compute the activations of the layer `layer` using `img` as input
    # shape: (layer_channels, layer_height, layer_width), the shape depends on the layer
    acts = get_layer(model, layer, img)[0]
    # shape: (layer_height, layer_width, layer_channels)
    acts = acts.permute(1, 2, 0)
    # shape: (layer_height*layer_width, layer_channels)
    acts = acts.view(-1, acts.shape[-1])
    acts_np = acts.cpu().numpy()
    nb_cells = acts.shape[0]

    # negative matrix factorization `NMF` is used to reduce the number
    # of channels to n_groups. This will be used as the following.
    # Each cell image in the grid is decomposed into a sum of
    # (n_groups+1) images. First, each cell has its own set of parameters
    #  this is what is called `cells_params` (see below). At the same time, we have
    # a of group of images of size 'n_groups', which also have their own image parametrized
    # by `groups_params`. The resulting image for a given cell in the grid
    # is the sum of its own image (parametrized by `cells_params`)
    # plus a weighted sum of the images of the group. Each each image from the group
    # is weighted by `groups[cell_index, group_idx]`. Basically, this is a way of having
    # the possibility to make cells with similar activations have a similar image, because
    # cells with similar activations will have a similar weighting for the elements
    # of the group.
    if n_groups > 0:
        reducer = ChannelReducer(n_groups, "NMF")
        groups = reducer.fit_transform(acts_np)
        groups /= groups.max(0)
    else:
        groups = np.zeros([])
    # shape: (layer_height*layer_width, n_groups)
    groups = torch.from_numpy(groups)

    # Parametrization of the images of the groups (we have 'n_groups' groups)
    groups_params, groups_image_f = param.fft_image(
        [n_groups, 3, cell_image_size, cell_image_size]
    )
    # Parametrization of the images of each cell in the grid (we have 'layer_height*layer_width' cells)
    cells_params, cells_image_f = param.fft_image(
        [nb_cells, 3, cell_image_size, cell_image_size]
    )

    # First, we need to construct the images of the grid
    # from the parameterizations

    def image_f():
        groups_images = groups_image_f()
        cells_images = cells_image_f()
        X = []
        for i in range(nb_cells):
            x = 0.7 * cells_images[i] + 0.5 * sum(
                groups[i, j] * groups_images[j] for j in range(n_groups)
            )
            X.append(x)
        X = torch.stack(X)
        return X

    # make sure the images are between 0 and 1
    image_f = param.to_valid_rgb(image_f, decorrelate=True)

    # After constructing the cells images, we sample randomly a mini-batch of cells
    # from the grid. This is to prevent memory overflow, especially if the grid
    # is large.
    def sample(image_f, batch_size):
        def f():
            X = image_f()
            inds = torch.randint(0, len(X), size=(batch_size,))
            inputs = X[inds]
            # HACK to store indices of the mini-batch, because we need them
            # in objective func. Might be better ways to do that
            sample.inds = inds
            return inputs

        return f

    image_f_sampled = sample(image_f, batch_size=batch_size)

    # Now, we define the objective function

    def objective_func(model):
        # shape: (batch_size, layer_channels, cell_layer_height, cell_layer_width)
        pred = model(layer)
        # use the sampled indices from `sample` to get the corresponding targets
        target = acts[sample.inds].to(pred.device)
        # shape: (batch_size, layer_channels, 1, 1)
        target = target.view(target.shape[0], target.shape[1], 1, 1)
        dot = (pred * target).sum(dim=1).mean()
        return -dot

    obj = objectives.Objective(objective_func)

    def param_f():
        # We optimize the parametrizations of both the groups and the cells
        params = list(groups_params) + list(cells_params)
        return params, image_f_sampled

    results = render.render_vis(
        model,
        obj,
        param_f,
        thresholds=(n_steps,),
        show_image=False,
        progress=True,
        fixed_image_size=cell_image_size,
    )

    # shape: (layer_height*layer_width, 3, grid_image_size, grid_image_size)
    imgs = image_f()
    imgs = imgs.cpu().data
    imgs = imgs[:, :, 2:-2, 2:-2]
    # turn imgs into a a grid
    grid = torchvision.utils.make_grid(imgs, nrow=int(np.sqrt(nb_cells)), padding=0)
    grid = grid.permute(1, 2, 0)
    grid = grid.numpy()
    render.show(grid)

    return imgs, grid