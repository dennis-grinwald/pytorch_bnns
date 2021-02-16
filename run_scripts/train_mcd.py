import os
import sys
import time
import json

import numpy as np
from tqdm import tqdm
from torch import nn, optim, device, cuda
    
from bayesian_activation_maximisation.src.get_datasets import *
from bayesian_activation_maximisation.src.bayesian_models.MCD_CNNs import * 
from bayesian_activation_maximisation.src.train_eval_fn import *

# SETUP #

file_path = './confs/training_conf.json'
with open(file_path, 'r') as j:
    json_file = json.loads(j.read())

xp_conf = json_file["mcd_training_conf"] 
xp_conf['training_stats'] = dict()

if not os.path.exists(xp_conf['path']+'/models/'):
    os.makedirs(xp_conf['path']+'/models/')
    os.makedirs(xp_conf['path']+'/results/')

device = device("cuda:0" if cuda.is_available() else "cpu")
print(f'Hardware accelerator: {device}')

# LOAD DATA #

trainloader, testloader, trainset, testset, classes = globals()['load_'+xp_conf['ds']](
    batch_size=xp_conf['batch_size'])

xp_conf['classes'] = classes

# TRAINING #

for p in xp_conf['ps']:

    xp_conf['training_stats'][f'p_{p}'] = dict()
    init_model_save_path = xp_conf['path'] + '/models/' + f"{xp_conf['model']}_p_" + str(p)
    model = globals()[xp_conf['model']](p=p).to(device)
     
    criterion = nn.CrossEntropyLoss().to(device)
    train_accs = []
    test_accs = []

    for lr in xp_conf['lrs']:
        
        xp_conf['training_stats'][f'p_{p}'][f'lr_{lr}'] = dict()
        lr_model_save_path = init_model_save_path + f'_lr_{lr}' + '.pt'

        for wd in xp_conf['wds']:

            print(f'Training {xp_conf["model"]}\nDropout rate p: {p}\nWeight decay: {wd}')

            xp_conf['training_stats'][f'p_{p}'][f'lr_{lr}'][f'wd_{wd}'] = dict()
            final_model_save_path = lr_model_save_path + f'_wd_{wd}' + '.pt'            
            
            #optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            lr_scheduler = getattr(optim.lr_scheduler, xp_conf['lr_scheduler'])(
                optimizer, step_size=round(xp_conf['epochs'] / 10), gamma=xp_conf['gamma'])

            _, train_acc, test_acc = train(model, trainloader, testloader, criterion, optimizer, 
                scheduler=lr_scheduler, epochs=xp_conf['epochs'], p=p, model_save_path=final_model_save_path, 
                device=device)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            xp_conf['training_stats'][f'p_{p}'][f'lr_{lr}'][f'wd_{wd}']['model_save_path'] = final_model_save_path
            xp_conf['training_stats'][f'p_{p}'][f'lr_{lr}'][f'wd_{wd}']['train_accs'] = np.concatenate(train_accs)
            xp_conf['training_stats'][f'p_{p}'][f'lr_{lr}'][f'wd_{wd}']['test_accs'] = np.concatenate(test_accs)

# SAVE RESULTS #

xp_save_path = xp_conf['path'] + '/results/' + f'xp_{xp_conf["model"]}_' + f'xp_{xp_conf["model"]}_' + '_'.join('_'.join(time.ctime().split(' ')).split(':'))
np.save(xp_save_path, xp_conf)