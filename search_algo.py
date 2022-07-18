"""Evolutionary Search Algorithm.  """
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

def evolution_search(search_space,
                     total_queries=100,
                     num_init=50,
                     k=20,
                     population_size=30,
                     tournament_size=10,
                     mutation_rate=1.0,
                     mutate_encoding='adj',
                     cutoff=0,
                     random_encoding='adj',
                     deterministic=True):

    data = search_space.generate_random_dataset(num=num_init, 
                                                random_encoding=random_encoding,
                                                deterministic_loss=deterministic)
    valid_loss = valid_loss()
    losses = [d[valid_loss] for d in data]
    query = num_init
    population = [i for i in range(min(num_init, population_size))]

    while query <= total_queries:

        # evolve the population by mutating the best architecture
        # from a random subset of the population
        sample = np.random.choice(population, tournament_size)
        best_index = sorted([(i, losses[i]) for i in sample], key=lambda i:i[1])[0][0]
        mutated = search_space.mutate_arch(data[best_index]['spec'],
                                           mutation_rate=mutation_rate, 
                                           mutate_encoding=mutate_encoding,
                                           cutoff=cutoff)
        arch_dict = search_space.query_arch(mutated, deterministic=deterministic)

        data.append(arch_dict)        
        losses.append(arch_dict[valid_loss])
        population.append(len(data) - 1)

        if len(population) >= population_size:
            worst_index = sorted([(i, losses[i]) for i in population], key=lambda i:i[1])[-1][0]
            population.remove(worst_index)

        if (query % k == 0):
            top_5_loss = sorted([d[valid_loss] for d in data])[:min(5, len(data))]
            print('evolution, query {}, top 5 losses {}'.format(query, top_5_loss))

        query += 1
    return data
