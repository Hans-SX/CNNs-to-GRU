import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.network import ConcatImg2Recur, ConcatImg2Transformer
from dataset.HFramesSet import Hframes_Interval
from utils import get_class, train_one_epoch

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--exp_name', type=str)
    # parser.add_argument('--model', type=str, default='ConcatImg2Transformer')
    # args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = r'results\sim_Netv3_resnet34_LSTM_lr1e-5_f128h128_MSELoss\cnn2gru_20251108_135918_41'
    testset = Hframes_Interval('test')
    testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=8)
    
    L1metric = nn.L1Loss()
    loss_func = nn.MSELoss()

    model = ConcatImg2Transformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()


    running_tloss = 0.0
    running_tl1 = 0.0
    with torch.no_grad():
        for i, tdata in enumerate(testloader):
            test_spa, test_ang, tlabels = tdata
            test_spa = test_spa.to(device)
            test_ang = test_ang.to(device)
            tlabels = tlabels.to(device).float()

            toutputs = model(test_spa, test_ang)
            tloss = loss_func(toutputs, tlabels.reshape(-1, 1))
            running_tloss += tloss
            tL1 = L1metric(toutputs, tlabels.reshape(-1, 1))
            running_tl1 += tL1
    
    avg_tloss = running_tloss / (i + 1)
    avg_tl1 = running_tl1 / (i + 1)
    print('Test LOSS {} L1 {}'.format(avg_tloss, avg_tl1))
    
