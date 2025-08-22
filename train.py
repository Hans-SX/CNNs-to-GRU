import argparse
import yaml
import os
from os.path import join
from datetime import datetime
import shutil

from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from model.network import ConcatedCNN2GRU
from dataset.HFramesSet import Hframes_Interval

start_time = datetime.now()

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

# Create saving directory
save_root = join(os.getcwd(), "results", args.exp, exp_name)
if not os.path.exists(save_root):
    os.makedirs(save_root)
    print(f"Create {save_root}")
exp_name = config['exp_name']
print(f"EXP: {exp_name}")

# Tensorboard
writer = SummaryWriter(join(os.getcwd(), "results", "logs", exp_name))

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
seq_len = config['sequenc_length']

cnn2gru = ConcatedCNN2GRU(spa_width, spa_length, ang_width, ang_length, f_size, h_size, sequence_length=seq_len)

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
# valset = Hframes_Interval('val')


# Dataloader
trainloader = DataLoader(trainset, config['batch_size'], num_workers=config['num_workers'])
# valloader = DataLoader(valset, config['batch_size'], num_workers=config['num_workers'])
print(f"Number of batch: {len(trainloader)}/epoch")

########################################
# Start training
########################################
losses = []
print("\n> Training")
for epoch in range(config["max_epoch"]):
    #! from here -------------------------
    # sf_lambda = torch.zeros((len(trainloader), config['batch_size'], config['seq_len']), dtype=torch.cfloat)
    sf_lambda = []
    out_lambda = []

    for iter_train, (labels, gate_seq) in enumerate(trainloader):
        optimizer.zero_grad()

        # Drop labels and gate_seq into device
        labels = labels.to(device)
        gate_seq = gate_seq.to(device)


        # Forward
        out_sf, h_t = gru_model(gate_seq)

        # Input of the fc_model: sequence gates, last step ASF, last step hidden features
        fc_input = torch.cat((gate_seq.view(gate_seq.shape[0], -1), out_sf[:, -1].view(-1, 1), h_t.view(h_t.shape[1], -1)), 1)
        # out_lambda in the same batch are the same, each batch only need to record one.
        out_lambda_iter = fc_model(fc_input)
        out_lambda.append(out_lambda_iter)

        # Backward
        sf_lambda.append(asf_from_lambda(out_lambda_iter, gate_seq, config['lambda_dim']).real)
        loss_asf, loss_lambda = criterion(out_sf, sf_lambda[iter_train], labels)
        if regular == 0:
            loss_regular = 0
        else:
            loss_regular = regular(out_lambda_iter, config['lambda_dim'])

        if noise_guess == True:
            loss_nguess = frob_norm_sq(noise_g, out_lambda_iter)
        else:
            loss_nguess = 0

        loss = config['weight_asf'] * loss_asf + config['weight_lambda'] * loss_lambda + config['weight_regular'] * loss_regular + config['weight_nguess'] * loss_nguess

        # Calculate gradient
        loss.backward()
        # Update parameters
        optimizer.step()

        # Recording training log (Tensorboard)
        if (iteration + 1) % config['log_iter'] == 0:
            print(f"Iteration: [{iter_train + 1:06d} / {len(trainloader):06d}] (Epoch: {epoch})")
            ########################################
            # Evaluation, each eiteration
            if config['eval'] == True:
                eval_val = frob_norm_sq(noise_u, out_lambda_iter)
                writer.add_scalar("eval", eval_val, iteration + 1)
            ########################################
            writer.add_scalar("loss", loss, iteration + 1)
            writer.add_scalar("loss_asf", loss_asf, iteration + 1)
            writer.add_scalar("loss_lambda", loss_lambda, iteration + 1)
            if noise_guess == True:
                writer.add_scalar("loss_nguess", loss_nguess, iteration + 1)
            if regular != 0:
                writer.add_scalar("loss_regular", loss_regular, iteration + 1)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], iteration + 1)
            print("memory usage percentage:", psutil.virtual_memory().percent)
        iteration += 1

    # Scheduler step in each epoch
    scheduler.step()

    losses.append(loss.detach())
    # Saving model
    torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'gru_state_dict': gru_model.state_dict(),
                'fc_state_dict': fc_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
    }, PJ(save_root, f"{epoch:04d}_{iteration:08d}_model.pt"))
    torch.save({
        "sf": torch.stack(sf_lambda),
        "lambda": torch.stack(out_lambda),
        "current_min_loss_epoch": losses.index(min(losses)),
        'losses': losses,
        "loss": loss
        }, PJ(save_root, f"{epoch:04d}_{iteration:08d}_sf_lambda.pt"))

    print(f"Saving model in {save_root} finished.\n")
    end_time = datetime.now()
    duration = end_time - start_time
    print('Duration: {}'.format(duration))
    writer.flush()

writer.close()
print("Min loss now is at the ", losses.index(min(losses)), "th epoch")
