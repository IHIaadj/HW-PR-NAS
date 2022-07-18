"""Encoding class & architecture."""
import numpy as np
import torch
import torch.nn as nn 
from gcn_encoder import GCNEncoder
from lstm_encoder import LSTMEncoder
from torchinfo import summary

class encoding(nn.Module):
    def __init__(self):
        self.gcn = GCNEncoder()
        self.lstm = LSTMEncoder()
        self.feature_extractor = FE()

    def forward(self, x):
        gcn_x = self.gcn(x)
        lstm_x = self.lstm(x)
        fe_x = self.feature_extractor(x)
        return [gcn_x,lstm_x],fe_x

class FE:
    def __init__(self, arch) -> None:
        self.arch = arch
        self.features_names = ["depth", "nb_conv", "FLOPs", "nb_param", "input_size", "1_channel", "last_channel", "nb_down_sample"]
        self.features = {}

    def extract_features(self):
        model_stats = summary(self.arch)
        summary_str = str(model_stats)

        return summary_str



    

