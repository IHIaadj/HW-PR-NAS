import numpy as np
import copy
import itertools
import random
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn 

def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian

    
class GCNEncoder(nn.Module):
    def __init__(self, adj, input_dim: int, output_dim: int, **kwargs):
        super(GCNEncoder, self).__init__()
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(0, 2).transpose(1, 2)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
        ax = self.laplacian @ inputs
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        outputs = torch.tanh(ax @ self.weights)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        outputs = outputs.transpose(0, 1)
        return outputs

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {
            "num_nodes": self._num_nodes,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
        }
