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

from bayes_act_max.bayesian_nn.utils import eval_curve_bnn
from bayes_act_max.bayesian_nn.bayesian_models.mcd_utils import *
from bayes_act_max.bayesian_nn.utils import *
from bayes_act_max.datasets.imagenet_labels import imagenet_labels

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="googlenet", type=str, help="The model used")

args = parser.parse_args()

# SETUP #
file_path = './confs/predictive_entropies.json'
with open(file_path, 'r') as j:
    xp_conf = json.loads(j.read())[args.model]

model_factors_path = '/home/dgrinwald/tools/curvature/factors/' + xp_conf['model'] + f'_{xp_conf["ds"]}_' + 'kfac.pth'

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
model = globals()[xp_conf['model']](pretrained=True).to(device)
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

mean_predictions, labels, stats_list = eval_curve_bnn(model, posterior_mean, seeds, dataset, inv_factors)

results = {
    'mean_predictions': mean_predictions,
    'labels': labels,
    'stats_list': stats_list
}

torch.save(results, xp_conf['results_path']+'xp_results.npy')