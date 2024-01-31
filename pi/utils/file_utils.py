# Created by jing at 31.01.24
import json
import torch

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
    print('Saved checkpoint file: {}'.format(filename))


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    print('Loaded checkpoint file: {}'.format(filename))
    return data


def save_pt(data, filename):
    torch.save(data, filename)
