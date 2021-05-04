import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import argparse
import copy

import torch
from torch import cuda, device
from torchvision.models import *
import torch.nn.functional as F

from curvature.sampling import invert_factors
from curvature import imagenet

from src.train_exp_utils.utils import eval_curve_bnn
from src.models.mcd_utils import *
from src.models.utils import sample_curve_network, get_network_activations_curve, get_sorted_channel_activations
from src.datasets.imagenet_labels import imagenet_labels
 
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="googlenet", type=str, help="The model used")

args = parser.parse_args()

# SETUP #
file = open("/home/dgrinwald/projects/bayesian_activation_maximisation/imagenet/imagenet_labels.txt", "r")
contents = file.read()
imgnet_classes = ast.literal_eval(contents)
file.close()

file_path = './confs/predictive_entropies_xp.json'
with open(file_path, 'r') as j:
    xp_conf = json.loads(j.read())[args.model]

results_path = xp_conf['results_path'] + f'/{xp_conf["ds"]}/' 
model_factors_path = '/home/dgrinwald/tools/curvature/factors/' + xp_conf['model'] + f'_{xp_conf["ds"]}_' + 'kfac.pth'

if not os.path.exists(results_path):
    os.makedirs(results_path)

# GET GPU # 
device = device("cuda:0" if cuda.is_available() else "cpu")
print(f'Hardware accelerator: {device}')

# SAMPLE THE NETWORKS #
# MAKE SURE WE ALWAYS SAMPLE THE SAME NETWORKS - SEED 42 #
np.random.seed(xp_conf['np_seed'])
samples = xp_conf['num_samples']
seeds = np.random.randint(0, 10000000, size=(samples, 2))

# LOAD BASE MODEL AND FACTORS #
print("Loading and inverting factors...")
model = globals()[xp_conf['model']](pretrained=True).to(device).eval()
factors = torch.load(model_factors_path)
pre_scale, scale, norm = xp_conf['pre_scale'], xp_conf['scale'], xp_conf['norm']
print(f'Prescale: {pre_scale}, scale: {scale}, norm: {norm}')
inv_factors = invert_factors(factors, norm, pre_scale * scale, xp_conf['estimator'])
posterior_mean = copy.deepcopy(model.state_dict())

# LOAD HIGH ENTROPY IMAGES
load_path = results_path + xp_conf['model'] + '_curve/' + f'{xp_conf["xp_name"]}_' + xp_conf['model'] + '_' +xp_conf['additional_name'] + '.npy'
xp = np.load(load_path, allow_pickle=True).item()

high_ent_imgs = xp['high_entropy_imgs']
high_ent_labels = xp['high_entropy_labels']

low_ent_imgs = xp['low_entropy_imgs']
low_ent_labels = xp['low_entropy_labels']

def make_predictions(img, label, inv_factors, posterior_mean, seeds):
  
  print(f'\nActual class: {imgnet_classes[label]}')
  
  for i, seed in enumerate(seeds):
    
    # SAMPLE A NETWORK #
    sample_curve_network(model, inv_factors, xp_conf['estimator'], posterior_mean, seed)
    pred = model(img.reshape((1,3,224,224)).to(device)).max(1)[1].detach().cpu().numpy()
    print(f'Net {i} predicts: {imgnet_classes[pred[0]]}')

idx = 1
img, label = high_ent_imgs[idx], high_ent_labels[idx]
make_predictions(img, label, inv_factors, posterior_mean, seeds)

# 2nd TASK FIND HIGHEST VARYING CHANNEL ACTIVATIONS #
net_acts = {f'net_{i}': get_network_activations_curve(model, inv_factors, xp_conf['estimator'], posterior_mean, seed, img, device) for i, seed in enumerate(seeds)}

sorted_channel_activations = get_sorted_channel_activations(net_acts)
tmp_layers = list(zip(*sorted_channel_activations[:xp_conf['no_channels']]))[0]
tmp_channels = list(zip(*sorted_channel_activations[:xp_conf['no_channels']]))[1]

sorted_channel_names = list(zip(tmp_layers, tmp_channels))

print(sorted_channel_names)
print(imgnet_classes[375])

#save_path = results_path + xp_conf['model'] + '_curve/' + f'feature_viz_' + xp_conf['model'] + '_' +xp_conf['additional_name'] + '.npy'
#np.save(save_path, net_acts)
