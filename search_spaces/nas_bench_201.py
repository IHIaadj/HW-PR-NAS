"""Load NAS-Bench-201 architectures."""
import numpy as np
import copy
import itertools
import random
import sys
import os
import pickle
import torch

INPUT = 'input'
OUTPUT = 'output'
OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
NUM_OPS = len(OPS)
OP_SPOTS = 6
LONGEST_PATH_LENGTH = 3

class Cell201:

    def __init__(self, string):
        self.string = string

    def get_string(self):
        return self.string

    def serialize(self):
        return {
            'string':self.string
        }

    def random_cell(cls):
        ops = []
        for i in range(OP_SPOTS):
            op = random.choice(OPS)
            ops.append(op)
        return {'string':cls.get_string_from_ops(ops)}

    def get_runtime(self, nasbench, dataset='cifar100'):
        return nasbench.query_by_index(index, dataset).get_eval('x-valid')['time']

    def get_val_loss(self, nasbench, deterministic=1, dataset='cifar100'):
        index = nasbench.query_index_by_arch(self.string)
        if dataset == 'cifar10':
            results = nasbench.query_by_index(index, 'cifar10-valid')
        else:
            results = nasbench.query_by_index(index, dataset)

        accs = []
        for key in results.keys():
            accs.append(results[key].get_eval('x-valid')['accuracy'])

        if deterministic:
            return round(100-np.mean(accs), 10)   
        else:
            return round(100-np.random.choice(accs), 10)

    def get_test_loss(self, nasbench, dataset='cifar100', deterministic=1):
        index = nasbench.query_index_by_arch(self.string)
        results = nasbench.query_by_index(index, dataset)

        accs = []
        for key in results.keys():
            accs.append(results[key].get_eval('ori-test')['accuracy'])

        if deterministic:
            return round(100-np.mean(accs), 4)   
        else:
            return round(100-np.random.choice(accs), 4)

    def get_op_list(self):
        # given a string, get the list of operations

        tokens = self.string.split('|')
        ops = [t.split('~')[0] for i,t in enumerate(tokens) if i not in [0,2,5,9]]
        return ops

    def get_num(self):
        # compute the unique number of the architecture, in [0, 15624]
        ops = self.get_op_list()
        index = 0
        for i, op in enumerate(ops):
            index += OPS.index(op) * NUM_OPS ** i
        return index

    