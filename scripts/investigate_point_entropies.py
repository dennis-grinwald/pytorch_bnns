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

from bayes_act_max.bayesian_nn.utils import eval_curve_bnn
from bayes_act_max.bayesian_nn.bayesian_models.mcd_utils import *
from bayes_act_max.bayesian_nn.utils import sample_curve_network, get_network_activations_curve, get_sorted_channel_activations
from bayes_act_max.datasets.imagenet_labels import imagenet_labels

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

# GET THE DATA #
print("Loading the data...")
img_size = 224
data_dir = '/home/dgrinwald/tools/curvature/datasets/imagenet/'
dataset = imagenet.imagenet(
    data_dir, img_size, 50, workers=2, splits='val', val_size=xp_conf['val_size'])

# LOAD HIGH ENTROPY IMAGES
load_path = results_path + xp_conf['model'] + '_curve/' + f'{xp_conf["xp_name"]}_' + xp_conf['model'] + '_' +xp_conf['additional_name'] + '.npy'
high_ent_imgs = np.load(load_path, allow_pickle=True).item()['high_entropy_imgs']
high_ent_labels = np.load(load_path, allow_pickle=True).item()['high_entropy_labels']

low_ent_imgs = np.load(load_path, allow_pickle=True).item()['low_entropy_imgs']
low_ent_labels = np.load(load_path, allow_pickle=True).item()['low_entropy_labels']

def make_predictions(img, label, inv_factors, posterior_mean, seeds):
  
  print(f'\nActual class: {imgnet_classes[label]}')
  
  for i, seed in enumerate(seeds[:10]):
    
    # SAMPLE A NETWORK #
    sample_curve_network(model, inv_factors, xp_conf['estimator'], posterior_mean, seed)

    pred = model(img.reshape((1,3,224,224)).to(device)).max(1)[1].detach().cpu().numpy()

    print(f'Net {i} predicts: {imgnet_classes[pred[0]]}')

idx = 1
img, label = high_ent_imgs[idx], high_ent_labels[idx]

make_predictions(img, label, inv_factors, posterior_mean, seeds)


 
