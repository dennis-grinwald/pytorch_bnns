import os
import sys
import json
import copy
import time

import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import torch

from torch import cuda, device
from curvature.sampling import invert_factors

from bayes_act_max.datasets.get_datasets import *
from bayes_act_max.bayesian_nn.utils import sample_curve_network, compute_attribution_map

import lucent
from lucent.util import set_seed

# Fix lucent seed
set_seed(42)

# SETUP #
file_path = './confs/compute_attribution_maps_inceptionv1.json'
with open(file_path, 'r') as j:
    xp_conf = json.loads(j.read())

if not os.path.exists(xp_conf['attr_maps_save_path']):
    os.makedirs(xp_conf['attr_maps_save_path'])

np.random.seed(xp_conf['np_seed'])
samples = xp_conf['num_nets']
seeds = np.random.randint(0, 10000000, size=(samples, 2))

# GET GPU # 
device = device("cuda:0" if cuda.is_available() else "cpu")
print(f'Hardware accelerator: {device}')

# Get data
trainloader, valloader, testloader, testset, idx_to_class, class_to_name = \
  globals()['load_'+xp_conf['ds']](
    batch_size=xp_conf['batch_size'])

# Load pretrained model and factors 
model_path = xp_conf['model_path']
factors_path = xp_conf['factors_path']

model = torch.load(model_path)
factors = torch.load(factors_path)
posterior_mean = copy.deepcopy(model.state_dict())

estimator = 'kfac'
inv_factors = invert_factors(factors, norm=1e3, scale=1e5, estimator='kfac')

# Load max entropy points
pred_dists = np.load(xp_conf['points_path'])
entropies = entropy(pred_dists.reshape(-1,200), axis=1)
print(entropies.shape)

# Compute max entropies
max_entropy_ids = np.argsort(entropies)[::-1]
min_entropy_ids = np.argsort(entropies)

# Highest entropy images
# total = 10
# fig, ax = plt.subplots(1, total, figsize=(30,15))
# for i in range(total):
#   img = np.transpose(testset[max_entropy_ids[i]][0], (1,2,0))
#   preprocessed_img = (img - img.min()) / (img.max() - img.min())
#   ax[i].imshow(preprocessed_img)
#   ax[i].set_title(f'ID: {max_entropy_ids[i]}, label: {class_to_name[idx_to_class[testloader.dataset[max_entropy_ids[i]][1]]]}')

# plt.savefig('./experiments/curve_inceptionv1/high_entropy_points/high_entropy_points.png')

# # Lowest entropy images
# total = 10
# fig, ax = plt.subplots(1, total, figsize=(30,15))
# for i in range(total):
#   img = np.transpose(testset[min_entropy_ids[i]][0], (1,2,0))
#   preprocessed_img = (img - img.min()) / (img.max() - img.min())
#   ax[i].imshow(preprocessed_img)
#   ax[i].set_title(f'ID: {min_entropy_ids[i]}, label: {class_to_name[idx_to_class[testloader.dataset[min_entropy_ids[i]][1]]]}')

# plt.savefig('./experiments/curve_inceptionv1/high_entropy_points/low_entropy_points.png')

# Loop
# Compute attribution maps for all images
for data_id in max_entropy_ids[:10]:

    if not os.path.exists(xp_conf['attr_maps_save_path']+f'{data_id}/'):
        os.makedirs(xp_conf['attr_maps_save_path']+f'{data_id}/')

    img = testset[data_id][0].reshape(1, 3, 224, 224)
    preprocessed_img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(np.transpose(preprocessed_img.reshape(3,224,224),(1,2,0)))
    fig_save_path = xp_conf['attr_maps_save_path']+f'/{data_id}/'+'original_img' 
    plt.savefig(fig_save_path)

    attr_maps_arr = []
    attr_maps_grid = []
    preds = []

    for i, seed in enumerate(seeds):
        
        # Sample net
        sample_curve_network(model, inv_factors, estimator, posterior_mean, seed)

        #Compute attribution map
        img = np.transpose(testset[data_id][0], (1,2,0))
        tmp_attr_map_arr, tmp_attr_map_grid = compute_attribution_map(img=img, model=model, 
                                                  cell_image_size=60, n_steps=1024,
                                                  n_groups=6, layer='mixed5b',
                                                  batch_size=64, device=device)
        # Save attribution maps
        attr_maps_arr.append(tmp_attr_map_arr)
        attr_maps_grid.append(tmp_attr_map_grid)

        img = testset[data_id][0].reshape(1, 3, 224, 224)

        # Compute network predictions
        pred = model(img.reshape(1, 3, 224, 224).to(device)).max(1)[1].detach().cpu().numpy()
        preds.append((pred[0], class_to_name[idx_to_class[pred[0]]]))

    xp_conf['attr_maps_arr'] = attr_maps_arr
    xp_conf['attr_maps_grid'] = attr_maps_grid
    xp_conf['preds'] = preds

    xp_save_path = xp_conf['attr_maps_save_path']+f'/{data_id}/'+f'xp_dict_'+'_'.join('_'.join(time.ctime().split(' ')).split(':'))
    np.save(xp_save_path, xp_conf)
