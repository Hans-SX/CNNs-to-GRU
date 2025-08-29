import argparse
import yaml
import os
from os.path import join
from datetime import datetime
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from model.network import ConcatedCNN2GRU
from dataset.HFramesSet import Hframes_Interval
from utils import train_one_epoch


########################################
# Environment and Experiment setting
########################################
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='exp')
args = parser.parse_args()

# Load experiment config
config_path = join("./config/", f"{args.exp}.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
exp_name = config['exp_name']
print(f"EXP: {exp_name}")

# Create saving directory
save_root = join(os.getcwd(), "results", exp_name)
if not os.path.exists(save_root):
    os.makedirs(save_root)
    print(f"Create {save_root}")

# Tensorboard
# writer = SummaryWriter(join(os.getcwd(), "results", exp_name, "logs"))
writer = SummaryWriter(join(os.getcwd(), "results/logs", exp_name))

# Saving config file
shutil.copy(config_path, join(save_root, f"{exp_name}.yaml"))

########################################
# Model
########################################

spa_width = config['spa_width']
spa_length = config['spa_length']
ang_width = config['ang_width']
ang_length = config['ang_length']
f_size = config['feature_size']
h_size = config['hidden_size']
seq_len = config['sequence_length']

cnn2gru = ConcatedCNN2GRU(spa_length, spa_width, ang_length, ang_width, f_size, h_size, sequence_length=seq_len)

########################################
# Loss function
########################################
loss_func = nn.L1Loss()

########################################
# Optimizer & Scheduler
########################################
optimizer = Adam(list(cnn2gru.parameters()), config['lr'], weight_decay=config['weight_decay'])

scheduler = lr_scheduler.MultiStepLR(
    optimizer, config["step_size"], config["gamma"])
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['Tmax'], eta_min=config['eta_min'])

########################################
# Data loader
########################################
# Dataset
trainset = Hframes_Interval()
valset = Hframes_Interval('val')


# Dataloader
trainloader = DataLoader(trainset, config['batch_size'], num_workers=config['num_workers'])
valloader = DataLoader(valset, config['batch_size'], num_workers=config['num_workers'])
print(f"Number of batch: {len(trainloader)}/epoch")

########################################
# Start training
########################################
print("\n> Training")

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    cnn2gru.train(True)
    avg_loss = train_one_epoch(epoch_number, writer, trainloader, optimizer, cnn2gru, loss_func)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    cnn2gru.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(valloader):
            vinputs, vlabels = vdata
            voutputs = cnn2gru(vinputs)
            vloss = loss_func(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    # writer.add_scalars('Training vs. Validation Loss',
    #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
    #                 epoch_number + 1)
    writer.add_scalar(f"train/avg_loss", avg_loss, epoch_number + 1)
    writer.add_scalar(f"val/avg_loss", avg_vloss, epoch_number + 1)

    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'cnn2gru_{}_{}'.format(timestamp, epoch_number)
        torch.save(cnn2gru.state_dict(), join(save_root, model_path))

    epoch_number += 1
