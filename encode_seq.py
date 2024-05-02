#! /usr/bin/env python3
"""
Converts structures into 3Di sequences.

echo 'd12asa_' | ./struct2states.py encoder_.pt states_.txt --pdb_dir data/pdbs --virt 270 0 2
"""

import numpy as np
import sys
import os.path
import argparse

import torch

# 50 letters (X/x are missing)
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz'

def distance_matrix(a, b):
    return np.sqrt(np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :])**2, axis=-1))
    # ab = a.dot(b.T)
    # a2 = np.square(a).sum(axis=1)
    # b2 = np.square(b).sum(axis=1)
    # d = np.sqrt(-2 * ab  + a2[:, np.newaxis] + b2)
    # return d

def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).detach().numpy()


def discretize(encoder, centroids, x):
    z = predict(encoder, x)
    print(z)
    return np.argmin(distance_matrix(z, centroids), axis=1)


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('encoder', type=str, help='a *.pt file')
    arg.add_argument('centroids', type=str, help='np.loadtxt')
    arg.add_argument('--pdb_dir', type=str, help='path to PDBs')
    arg.add_argument('--virt', type=float, nargs=3, help='virtual center')
    arg.add_argument('--invalid-state', type=str, help='for missing coords.',
        default='X')
    arg.add_argument('--exclude-feat', type=int, help='Do not calculate feature no.',
        default=None)
    args = arg.parse_args()


    encoder = torch.load(args.encoder)
    centroids = np.loadtxt(args.centroids)

    feat = np.load("vaevq_data.npy")

    for i in feat:
        print(discretize(encoder, centroids, feat))
        break
        
    # valid_states = discretize(encoder, centroids, feat)
    # print(len(valid_states))
    # print(valid_states.shape)
    # states = np.full(len(feat), -1)
    # states = valid_states

    # print(''.join([LETTERS[state] if state != -1 else args.invalid_state
    #     for state in states]))