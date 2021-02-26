import os
import sys
import json
import time

import numpy as np
from scipy.stats import entropy

import torch
from torch import cuda, device
import torch.nn.functional as F

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
