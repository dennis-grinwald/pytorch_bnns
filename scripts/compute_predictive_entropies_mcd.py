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

from bayesian_activation_maximisation.src.get_datasets import *
from bayesian_activation_maximisation.src.bayesian_models.mcd_utils import *

def y_pred_distribution(seeds, model, dataloader):

  start_time = time.time()

  batch_size = dataloader.batch_size
  num_labels = len(np.unique(dataloader.dataset.targets))
  y_pred_dists = np.zeros(([len(dataloader), batch_size, num_labels]))
  num_nets = seeds.shape[0] 

  for j, seed in enumerate(seeds):
      print(f'Network {j}')

      # Sample a network
      freeze(model, seed)  

      for i, (img, _) in enumerate(dataloader):
        
        img = img.to(device)

        # Do prediction
        y_pred = torch.exp(F.log_softmax(model(img).to(device), dim=1))
        y_pred = (y_pred == y_pred.max(dim=1, keepdim=True)[0]) * y_pred

        y_pred_np = y_pred.cpu().detach().numpy()
        y_pred_dists[i] += y_pred_np / num_nets 

  entropies = entropy(y_pred_dists.reshape(-1,num_labels), axis=1)
  
  end_time = time.time()

  print(f"Total time: {end_time-start_time}")

  return y_pred_dists, entropies

# SETUP #
file_path = './confs/predictive_entropies.json'
with open(file_path, 'r') as j:
    xp_conf = json.loads(j.read())

if not os.path.exists(xp_conf['base_path_points']):
    os.makedirs(xp_conf['base_path_points'])

# GET GPU # 
device = device("cuda:0" if cuda.is_available() else "cpu")
print(f'Hardware accelerator: {device}')

# Sample 1000 Bayesian Neural Networks
# Make sure we always sample the same networks - seed 42
np.random.seed(xp_conf['np_seed'])
samples = xp_conf['num_samples']
seeds = np.random.randint(0, 10000000, size=(samples, 2))

# LOAD BASE MODEL #
model_name = xp_conf["model_name"]
model_path = xp_conf["base_path_models"]+model_name
model = torch.load(model_path)

# GET THE DATA #
trainloader, testloader, trainset, testset, classes = globals()['load_'+xp_conf['ds']](
    batch_size=xp_conf['batch_size'])

# COMPUTE PREDICTIVE ENTROPIES #
y_pred_dists, entropies = y_pred_distribution(seeds[0:100], model, testloader)

xp_conf['y_pred_dists'] = y_pred_dists 
xp_conf['entropies'] = entropies

# SAVE RESULTS
save_path = xp_conf['base_path_points'] + xp_conf['xp_name'] + '_high_ent_points_' \
            + xp_conf['model_name']

np.save(save_path, xp_conf)





