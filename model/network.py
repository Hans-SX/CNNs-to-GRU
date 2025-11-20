import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34


class Net_v3(nn.Module):
    def __init__(self, output_size=16):
        super().__init__()
        # self.cnn = resnet50()
        self.cnn = resnet34()
        # Change input conv layer to accept 1 channel images
        self.cnn.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        # Change output layer to output_size
        # self.cnn.fc = nn.Linear(2048, output_size)
        self.cnn.fc = nn.Linear(512, output_size)

    def forward(self, x):
        out = self.cnn(x)
        return out


class ConcatImg2Recur(nn.Module):
    def __init__(self, feature_size, img_model, recur_model, device, hidden_size=64, num_layers=1, sequence_length=100):
        super(ConcatImg2Recur, self).__init__()
        self.num_layers = num_layers
        self.hidden_size  = hidden_size
        self.device = device

        self.spa_cnn = img_model(feature_size)
        self.ang_cnn = img_model(feature_size)
        self.recur = recur_model(2 * feature_size, hidden_size, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1))

    def forward(self, spa, ang):
        """ spa: (B, seq_len, 1, 140, 100), ang: (B, seq_len, 1, 125, 125) """

        spa_feature = list()
        ang_feature = list()
        for i in range(spa.shape[1]):
            spa_feature.append(self.spa_cnn(spa[:, i]))
            ang_feature.append(self.ang_cnn(ang[:, i]))

        spa_feature = torch.stack(spa_feature, dim=1)  # (B, seq_len, feature_size)
        ang_feature = torch.stack(ang_feature, dim=1)  # (B, seq_len, feature_size)
        features = torch.cat((spa_feature, ang_feature), dim=2)  # (B, seq_len, 2*feature_size)

        # features = torch.cat((torch.cat(spa_feature, dim=0),
        #                       torch.cat(ang_feature, dim=0)), dim=1).reshape(spa.shape[0], spa.shape[1], -1)
        if type(self.recur) == nn.LSTM:
            h0 = torch.zeros(self.num_layers, features.shape[0], self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers, features.shape[0], self.hidden_size).to(self.device)
            init0 = (h0, c0)
        elif type(self.recur) == nn.GRU:
            init0 = torch.zeros(self.num_layers, features.shape[0], self.hidden_size).to(self.device)

        out,_ = self.recur(features, init0)      # out: (BS, L, H_out)
        out = out[:, -1]
        out = self.fc1(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [T, D]

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # [1, T, D] broadcast


class ConcatImg2Transformer(nn.Module):
    def __init__(self,
                 feature_size, img_model, device, hidden_size=64,
                 num_layers=1, scale=7):
        super().__init__()
        self.scale = scale
        self.num_layers = num_layers
        self.hidden_size  = hidden_size
        self.device = device

        self.cnn = img_model(feature_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2*feature_size, nhead=8, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc1 = nn.Sequential(
            nn.Linear(2*feature_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1))

        # Theoretically, not necessary
        self.pos_enc = None
        # self.pos_enc = PositionalEncoding(d_model=2*feature_size, max_len=100)

    def forward_visual_features(self, spa, ang):
        Bs, Ts, Cs, Hs, Ws = spa.shape
        Ba, Ta, Ca, Ha, Wa = ang.shape

        spa = spa.view(Bs*Ts, Cs, Hs, Ws)
        ang = ang.view(Ba*Ta, Ca, Ha, Wa)

        spa_feature = self.cnn(spa).view(Bs, Ts, -1)
        ang_feature = self.cnn(ang).view(Ba, Ta, -1)

        features = torch.cat((spa_feature, ang_feature), dim=2)  # (B, seq_len, 2*feature_size)
        return features

    def forward(self, spa, ang):
        """ spa: (B, seq_len, 1, 140, 100), ang: (B, seq_len, 1, 125, 125) """

        features = self.forward_visual_features(spa, ang)   # (B, seq_len, 2*feature_size)

        if self.pos_enc:
            features = self.pos_enc(features)
        features = self.transformer_encoder(features)

        out = features[:, -1]   # Last one looks better than average pooling (.mean(dim=1))
        out = self.fc1(out)
        if self.scale:
            out = torch.tanh(out) * self.scale
        return out
