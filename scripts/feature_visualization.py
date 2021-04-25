import os
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import argparse

import torch
from torch import cuda, device
from torchvision.models import *
import torch.nn.functional as F

from curvature.sampling import invert_factors
from curvature import imagenet

import lucent
from lucent import util
from lucent.optvis import render, param, transform, objectives

from bayes_act_max.bayesian_nn.utils import eval_curve_bnn
from bayes_act_max.bayesian_nn.bayesian_models.mcd_utils import *
from bayes_act_max.bayesian_nn.utils import *
from bayes_act_max.datasets.imagenet_labels import imagenet_labels

# SET LUCENT SEED
util.set_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="googlenet", type=str, help="The model used")

args = parser.parse_args()

# SETUP #
file_path = './confs/predictive_entropies_xp.json'
with open(file_path, 'r') as j:
    xp_conf = json.loads(j.read())[args.model]

results_path = xp_conf['results_path'] + f'/{xp_conf["ds"]}/' + f'{args.model}_curve/imgs/' 
model_factors_path = '/home/dgrinwald/tools/curvature/factors/' + xp_conf['model'] + f'_{xp_conf["ds"]}_' + 'kfac.pth'

if not os.path.exists(results_path):
    os.makedirs(results_path)

# GET GPU # 
device = device("cuda:0" if cuda.is_available() else "cpu")
print(f'Hardware accelerator: {device}')

# SAMPLE THE NETWORKS
# Make sure we always sample the same networks - seed 42
np.random.seed(xp_conf['np_seed'])
samples = xp_conf['num_samples']
seeds = np.random.randint(0, 10000000, size=(samples, 2))

# LOAD BASE MODEL AND FACTORS #
print("Loading and inverting factors...")
model = globals()[xp_conf['model']](pretrained=True).to(device).eval()
factors = torch.load(model_factors_path) 

pre_scale, scale, norm = xp_conf['pre_scale'], xp_conf['scale'], xp_conf['norm']

inv_factors = invert_factors(factors, norm, pre_scale * scale, xp_conf['estimator'])
posterior_mean = copy.deepcopy(model.state_dict())

def visualize_neuron_dif_seeds(model, seeds, layer, channel):
    
    fig, ax = plt.subplots(1, seeds.shape[0] + 1, figsize=(30,10))

    # Mean network Image
    act_max = render.render_vis(model, f'{layer}:'+f'{str(channel)}')
    ax[0].imshow(act_max[0][0])
    ax[0].set_title(f'Mean network')

    for i, seed in enumerate(seeds, start=1):
        
        sample_curve_network(model, inv_factors, xp_conf['estimator'], posterior_mean, seed)
        
        act_max = render.render_vis(model, f'{layer}:'+f'{str(channel)}')
        ax[i].imshow(act_max[0][0])
        ax[i].set_title(f'Net:{i}')

    plt.savefig(results_path + f'feature_viz_{layer}_{channel}_dif_seeds.png')

def visualize_neuron_dif_vars(model, seeds, layer, channel, vars):
    
    fig, ax = plt.subplots(1, len(vars) + 1, figsize=(30,10))

    # Mean network Image
    act_max = render.render_vis(model, f'{layer}:'+f'{str(channel)}')
    ax[0].imshow(act_max[0][0])
    ax[0].set_title(f'Mean network')

    for i, v in enumerate(vars, start=1):
        print(seeds[0])
        sample_curve_network(model, inv_factors, xp_conf['estimator'], posterior_mean, seeds[0], var_hp=v)
        act_max = render.render_vis(model, f'{layer}:'+f'{str(channel)}')
        ax[i].imshow(act_max[0][0])
        ax[i].set_title(f'Net:{i}')

    plt.savefig(results_path + 'feature_viz_dif_vars.png')

if args.model == 'densenet121':
    visualize_neuron_dif_seeds(model, seeds, 'features_denseblock4_denselayer16_conv2', 2)

elif args.model == 'googlenet':
    visualize_neuron_dif_seeds(model, seeds, 'inception5b_branch3_1', 96)
    visualize_neuron_dif_seeds(model, seeds, 'inception5b', 864)
    visualize_neuron_dif_seeds(model, seeds, 'fc', 375)
    visualize_neuron_dif_seeds(model, seeds, 'inception5b_branch2_1', 120)
    visualize_neuron_dif_seeds(model, seeds, 'inception5b_branch2_1', 120)

elif args.model == 'resnet50':
    #visualize_neuron_dif_seeds(model, seeds, 'labels', 3)
    visualize_neuron_dif_vars(model, seeds[:1], "labels", 9, [1, 5, 10, 15, 20])
