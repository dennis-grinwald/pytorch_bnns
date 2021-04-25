import os
import sys
import json
import time
import argparse
import copy

import numpy as np
from scipy.stats import entropy

import torch
from torch import cuda, device
from torchvision.models import *
import torch.nn.functional as F

from curvature.sampling import invert_factors
from curvature import imagenet

from bayes_act_max.datasets.get_datasets import *
from bayes_act_max.bayesian_nn.utils import eval_curve_bnn
from bayes_act_max.bayesian_nn.bayesian_models.mcd_utils import *
from bayes_act_max.bayesian_nn.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="googlenet", type=str, help="The model used")

args = parser.parse_args()

# SETUP #
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

# COMPUTE PREDICTIONS
mean_predictions, labels, stats_list = eval_curve_bnn(model, posterior_mean, seeds, dataset, inv_factors)

# STORE PREDICTIVE ENTROPIES #
xp_conf['mean_predictions'] = mean_predictions
xp_conf['entropies'] = stats_list['ent'][-1]

# FIND HIGHEST ENTROPY POINTS(MAX TO MIN) #
xp_conf['sorted_entropy_ids'] = np.argsort(xp_conf['entropies'])[::-1]

# STORE LOW ENTROPY IMAGES(DEFAULT: 10 IMAGES) #
high_entropy_images = [dataset.dataset[i][0] for i in xp_conf['sorted_entropy_ids'][:10]]
high_entropy_labels = [dataset.dataset[i][1] for i in xp_conf['sorted_entropy_ids'][:10]]
xp_conf['high_entropy_imgs'] = high_entropy_images
xp_conf['high_entropy_labels'] = high_entropy_labels

# STORE LOW ENTROPY IMAGES(DEFAULT: 10 IMAGES) #
low_entropy_images = [dataset.dataset[i][0] for i in xp_conf['sorted_entropy_ids'][::-1][:10]]
low_entropy_labels = [dataset.dataset[i][1] for i in xp_conf['sorted_entropy_ids'][::-1][:10]]
xp_conf['low_entropy_imgs'] = low_entropy_images
xp_conf['low_entropy_labels'] = low_entropy_labels

# SAVE RESULTS #
save_path = results_path + xp_conf['model'] + '_curve/' + f'{xp_conf["xp_name"]}_' + xp_conf['model'] + '_' +xp_conf['additional_name'] + '.npy'
np.save(save_path, xp_conf)